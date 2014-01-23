/*
 * Detector.cpp
 *
 *  Created on: Aug 23, 2013
 *      Author: coert
 */

#include "Detector.h"

using namespace cv;
using namespace std;
using namespace boost;
using namespace nl_uu_science_gmt;

namespace nl_uu_science_gmt
{

string Detector::ConfigPath = "../data/config.xml";
QueryXML* Detector::_Config = NULL;
string Detector::StorageExt = "xml";
string Detector::ImageExt = "png";

struct ResultLocation
{
	int layer;
	int offset;
	double score;
};

struct descending
{
	inline bool operator()(const ResultLocation& left, const ResultLocation& right)
	{
		return (left.score < right.score);
	}
};

struct ascending
{
	inline bool operator()(const ResultLocation& left, const ResultLocation& right)
	{
		return (left.score > right.score);
	}
};

Detector::Detector(const std::string &qif) :
		_pos_amount(Detector::cfg()->getValue<int>("settings/images@amount")), _target_width(
				Detector::cfg()->getValue<int>("settings/images@width")), _posneg_factor(
				Detector::cfg()->getValue<int>("settings/images@factor")), _seed(
				Detector::cfg()->getValue<int>("settings@seed")), _disp(
				Detector::cfg()->getValue<int>("settings/images/examples@size")), _max_count(
				Detector::cfg()->getValue<size_t>("settings/svm/params/max_count")), _epsilon(
				Detector::cfg()->getValue<double>("settings/svm/params/epsilon")), _max_image_size(
				Detector::cfg()->getValue<int>("settings/images/test/max_size")), _layer_scale_interval(
				Detector::cfg()->getValue<int>("settings/images/test/pyramid@layer_interval")), _initial_threshold(
				Detector::cfg()->getValue<int>("settings/images/test@threshold")), _overlap_threshold(
				Detector::cfg()->getValue<double>("settings/images/test/nms@threshold")), _gt_accuracy(
				Detector::cfg()->getValue<double>("settings/images/test/ground_truth@accuracy")), _do_equalizing(
				Detector::cfg()->getValue<bool>("settings/features@equalize")), _do_whitening(
				Detector::cfg()->getValue<bool>("settings/features@whiten")), _query_image_file(qif)
{
	assert(_max_count > 0);
	assert(_epsilon > 0);

	if (_seed > 0)
		srand(_seed);
	else
		srand((int64) cvGetTickCount());

	_layer_scale_factor = pow((double) 2, 1 / (double) _layer_scale_interval);
}

Detector::~Detector()
{
}

/*
 * Read positive image files (and shuffle)
 */
void Detector::readPosFilelist(vector<string> &pos_files)
{
	const string pos_path = Detector::cfg()->getValue<string>("settings/images/pos/path");
	const string dirs_regx = Detector::cfg()->getValue<string>("settings/images/pos/path@regx");
	const string mask_regx = Detector::cfg()->getValue<string>("settings/images/pos/files@regx");

	cout << "Read positive file names" << endl;
	vector<string> pos_directories;
	FileIO::getDirectory(pos_path, pos_directories, dirs_regx);

	vector<double> fps;
	int index = 0;
	int64 t0 = Utility::get_time_curr_tick();
	for (size_t d = 0; d < pos_directories.size(); ++d)
	{
		string directory = pos_directories[d];
		vector<string> d_files;
		string full_path = pos_path + directory + "/";
		assert(FileIO::isDirectory(full_path));
		FileIO::getDirectory(full_path, d_files, mask_regx, full_path);
		pos_files.insert(pos_files.end(), d_files.begin(), d_files.end());

		string etf = Utility::show_fancy_etf(index, pos_directories.size(), 100, t0, fps);
		if (!etf.empty()) cout << etf << endl;

		++index;
	}

	assert(pos_files.size() > 0);
	random_shuffle(pos_files.begin(), pos_files.end());
}

/*
 * Read negative image files (and shuffle)
 */
void Detector::readNegFilelist(vector<string> &neg_files)
{
	cout << "Read negative file names" << endl;
	const string neg_path = Detector::cfg()->getValue<string>("settings/images/neg/path");
	const string files_regx = Detector::cfg()->getValue<string>("settings/images/neg/files@regx");
	FileIO::getDirectory(neg_path, neg_files, files_regx, neg_path);
	assert(!neg_files.empty());
	random_shuffle(neg_files.begin(), neg_files.end());
}

/*
 * Read positive image data
 */
void Detector::readPosData(const std::vector<std::string> &pos_train, cv::Mat &pos_data)
{
	assert(!pos_train.empty());

	const int x1 = Detector::cfg()->getValue<int>("settings/features/crop/x1");
	const int y1 = Detector::cfg()->getValue<int>("settings/features/crop/y1");
	const int x2 = Detector::cfg()->getValue<int>("settings/features/crop/x2");
	const int y2 = Detector::cfg()->getValue<int>("settings/features/crop/y2");

	Mat image = imread(pos_train.front(), CV_LOAD_IMAGE_GRAYSCALE);
	const int width = _target_width;
	const int height = width * double(image.rows / (double) image.cols);
	if (_model_size.area() == 0) _model_size = Size(width, height);

	cout << "Read positive data (" << pos_train.size() << ")" << endl;
	Mat mv_pos_data, sv_pos_data;
	int index = 0;
	vector<double> fps;
	int64 t0 = Utility::get_time_curr_tick();
	Mat pos_sum;

	for (size_t f = 0; f < pos_train.size(); ++f)
	{
		string file = pos_train[f];
		image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		Rect rect = Rect(Point(x1, y1), Point(image.cols - x2, image.rows - y2));
		assert(rect.area() < image.size().area());
		Mat features = image(rect);
		resize(features, features, _model_size);

		if (pos_sum.empty())
		{
			Mat sz_imF;
			features.convertTo(sz_imF, CV_64F);
			pos_sum.push_back(sz_imF);
		}
		else
		{
			Mat sz_imF;
			features.convertTo(sz_imF, CV_64F);
			pos_sum += sz_imF;
		}

		Mat features1d = features.reshape(1, 1);
		Mat features1dF;
		features1d.convertTo(features1dF, CV_64F);

		Mat mean, stddev;
		meanStdDev(features1dF, mean, stddev);
		Mat mean_line(1, features1dF.cols, CV_64F);
		mean_line = mean.at<double>(0, 0);
		Mat std_line(1, features1dF.cols, CV_64F);
		std_line = stddev.at<double>(0, 0);
		mv_pos_data.push_back(mean_line);
		sv_pos_data.push_back(std_line);

		pos_data.push_back(features1dF);

		string etf = Utility::show_fancy_etf(index, pos_train.size(), 10, t0, fps);
		if (!etf.empty()) cout << etf << endl;

		++index;
	}

	if (_do_equalizing)
	{
		pos_data = pos_data - mv_pos_data;
		pos_data = pos_data / sv_pos_data;
	}

	// This is the mean model
	Mat pos_sum1dF = pos_sum.reshape(1, 1);
	_pos_sumF = pos_sum1dF / pos_data.rows;

	normalize(pos_sum1dF.reshape(1, height), pos_sum, 255, 0, NORM_MINMAX);
	pos_sum.convertTo(_pos_sum8U, CV_8U);
}

/*
 * Read negative image data
 */
void Detector::readNegData(const std::vector<std::string> &neg_train, cv::Mat &neg_data)
{
	assert(!neg_train.empty());

	Mat mv_neg_data, sv_neg_data;

	double factor = _pos_amount * _posneg_factor / (double) neg_train.size();
	int fpnt = ceil(MAX(factor, 1));  //guess we should only train on super models from here on... ;)

	cout << "Read negative data (" << neg_train.size() << ")" << endl;
	vector<double> fps;
	int index = 0;
	int64 t0 = Utility::get_time_curr_tick();
	Mat neg_sum;
	for (size_t f = 0; f < neg_train.size(); ++f)
	{
		string file = neg_train[f];
		Mat image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);

		for (int i = 0; i < fpnt; ++i)
		{
			int x = 0 + (rand() % ((image.cols - _model_size.width) - 0));
			int y = 0 + (rand() % ((image.rows - _model_size.height) - 0));
			assert(x + _model_size.width < image.cols);
			assert(y + _model_size.height < image.rows);
			Mat features = image(Rect(Point(x, y), _model_size)).clone();

			if (neg_sum.empty())
			{
				Mat sz_imF;
				features.convertTo(sz_imF, CV_64F);
				neg_sum.push_back(sz_imF);
			}
			else
			{
				Mat sz_imF;
				features.convertTo(sz_imF, CV_64F);
				neg_sum += sz_imF;
			}

			Mat features1d = features.reshape(1, 1);
			Mat features1dF;
			features1d.convertTo(features1dF, CV_64F);

			Mat mean_val, stddev_val;
			meanStdDev(features1dF, mean_val, stddev_val);
			Mat mean_line(1, features1dF.cols, CV_64F);
			mean_line = mean_val.at<double>(0, 0);
			Mat std_line(1, features1dF.cols, CV_64F);
			std_line = stddev_val.at<double>(0, 0);
			mv_neg_data.push_back(mean_line);
			sv_neg_data.push_back(std_line);

			neg_data.push_back(features1dF);

			string etf = Utility::show_fancy_etf(index, neg_train.size() * fpnt, _posneg_factor * 10, t0, fps);
			if (!etf.empty()) cout << etf << endl;

			++index;
		}
	}

	if (_do_equalizing)
	{
		neg_data = neg_data - mv_neg_data;
		neg_data = neg_data / sv_neg_data;
	}

	normalize(neg_sum, neg_sum, 255, 0, NORM_MINMAX);
	neg_sum.convertTo(_neg_sum8U, CV_8U);
}

/**
 * Creates a feature pyramid of several layers, each layer containing a smaller
 * representation of the image
 *
 * The max_layers is given by:
 * 		1 + floor(
 * 					log(MIN(image.rows, image.cols) / (double) 2 * _target_width) /
 * 										log(_layer_scale_factor)
 * 				);
 *
 * The amount of layers in the pyramid is given by:
 * 		MAX(max_layers, _layer_scale_interval) + _layer_scale_interval;
 *
 * Correct implementation causes the layer pyramid to halve its size every
 * _layer_scale_interval layers. So if _layer_scale_interval = 10,
 * and the image size is 200x100, then
 * layer 0 is 200x100
 * layer 1 is ...
 * layer 2 is ...
 * .
 * .
 * layer 10 is 100x50
 * .
 * .
 * layer 20 is 50x25
 * .
 * .
 *
 * The layers in between are linearly resized by a factor, in between the scale interval layers.
 */
void Detector::createPyramid(const Mat &image, vector<Mat*> &pyramid)
{
	pyramid.push_back(new Mat(image));
}

void Detector::run()
{
	assert(FileIO::isFile(_query_image_file));

	////////////////////// Read pos and neg examples ////////////////////////////
	vector<string> pos_files, neg_files;
	readPosFilelist(pos_files);
	readNegFilelist(neg_files);

	assert((int ) pos_files.size() > _pos_amount);
	assert((int ) neg_files.size() > 0);
	int neg_amount = MIN((int ) neg_files.size() / 2, _pos_amount * _posneg_factor / 2);

	cout << "Total images, pos:" << pos_files.size() << ", neg:" << neg_files.size() << endl;

	vector<string> pos_train(pos_files.begin(), pos_files.begin() + _pos_amount);
	vector<string> pos_val(pos_files.begin() + _pos_amount,
			pos_files.begin() + MIN(2 * _pos_amount, (int ) pos_files.size() - _pos_amount));
	vector<string> neg_train(neg_files.begin(), neg_files.begin() + neg_amount);
	vector<string> neg_val(neg_files.begin() + neg_amount,
			neg_files.begin() + MIN(2 * neg_amount, (int ) neg_files.size() - neg_amount));

	cout << "Positive training:" << pos_train.size() << ", validation:" << pos_val.size() << endl;
	cout << "Negative training:" << neg_train.size() << ", validation:" << neg_val.size() << endl;

	Mat pos_tmp_im = imread(pos_files.front(), CV_LOAD_IMAGE_GRAYSCALE);
	int width = _target_width;
	int height = width * double(pos_tmp_im.rows / (double) pos_tmp_im.cols);
	const Size sz_dim = Size(width, height);
	Mat pos_tmp_sz_im;
	resize(pos_tmp_im, pos_tmp_sz_im, sz_dim);

	Mat pos_train_data, neg_train_data;
	cout << endl << "line:" << __LINE__ << ") Read training images" << endl;
	cout << "==============================" << endl;
	readPosData(pos_train, pos_train_data);
	readNegData(neg_train, neg_train_data);
	/////////////////////////////////////////////////////////////////////////////

	/////////////////////////// Whitening transformation ////////////////////////
	Mat whitened_pos_data, whitened_neg_data;
	if (_do_whitening)
	{
		Utility::whiten(pos_train_data, _model_size, whitened_pos_data);
		Utility::whiten(neg_train_data, _model_size, whitened_neg_data);
	}
	/////////////////////////////////////////////////////////////////////////////

	////////////////////// Show pos and neg examples ////////////////////////////
	const int canvas_total = MIN(_disp, MIN(neg_train_data.rows, pos_train_data.rows));
	const int canvas_cols = sqrt((double) canvas_total);

	Mat pos_tmp_sz_im_canvas, neg_tmp_sz_im_canvas;
	Utility::get_images_canvas(canvas_cols, pos_train_data, sz_dim, pos_tmp_sz_im_canvas);
	Utility::get_images_canvas(canvas_cols, neg_train_data, sz_dim, neg_tmp_sz_im_canvas);

	if (_do_whitening)
	{
		Mat whitened_pos_tmp_sz_im_canvas;
		Utility::get_images_canvas(canvas_cols, whitened_pos_data, sz_dim, whitened_pos_tmp_sz_im_canvas);
		Mat line = Mat::zeros(Size(2, pos_tmp_sz_im_canvas.rows), pos_tmp_sz_im_canvas.type());
		hconcat(pos_tmp_sz_im_canvas, line, pos_tmp_sz_im_canvas);
		hconcat(pos_tmp_sz_im_canvas, whitened_pos_tmp_sz_im_canvas, pos_tmp_sz_im_canvas);
	}
	/////////////////////////////////////////////////////////////////////////////

	///////////// Build train / val datasets and display stuff  /////////////////
	Mat train_data;
	train_data.push_back(_do_whitening ? whitened_pos_data : pos_train_data);
	train_data.push_back(_do_whitening ? whitened_neg_data : neg_train_data);

	Mat pos_labels = Mat(pos_train_data.rows, 1, CV_32S, Scalar::all(1));
	Mat neg_labels = Mat(neg_train_data.rows, 1, CV_32S, Scalar::all(-1));

	Mat labels;
	labels.push_back(pos_labels);
	labels.push_back(neg_labels);

	cout << endl << "line:" << __LINE__ << ") Read validation images" << endl;
	cout << "==============================" << endl;
	Mat pos_val_data, neg_val_data;
	readPosData(pos_val, pos_val_data);
	readNegData(neg_val, neg_val_data);

	Mat whitened_pos_val_data, whitened_neg_val_data;
	if (_do_whitening)
	{
		Utility::whiten(pos_val_data, _model_size, whitened_pos_val_data);
		Utility::whiten(neg_val_data, _model_size, whitened_neg_val_data);
	}

	Mat val_data;
	val_data.push_back(_do_whitening ? whitened_pos_val_data : pos_val_data);
	val_data.push_back(_do_whitening ? whitened_neg_val_data : neg_val_data);

	cout << "Show windows" << endl;
	namedWindow("Pos examples", CV_WINDOW_KEEPRATIO);
	namedWindow("Neg examples", CV_WINDOW_KEEPRATIO);
	namedWindow("Pos mean image", CV_WINDOW_KEEPRATIO);
	namedWindow("Neg mean image", CV_WINDOW_KEEPRATIO);
	imshow("Pos examples", pos_tmp_sz_im_canvas);
	imshow("Neg examples", neg_tmp_sz_im_canvas);
	imshow("Pos mean image", _pos_sum8U);
	imshow("Neg mean image", _neg_sum8U);
	cout << "Press a key to continue" << endl;
	waitKey();
	namedWindow("Model", CV_WINDOW_KEEPRATIO);

	Mat val_labels(pos_val_data.rows, 1, CV_32S, Scalar::all(1));
	val_labels.push_back(Mat(neg_val_data.rows, 1, CV_32S, Scalar::all(-1)));
	Mat val_gnd = ((val_labels > 0) / 255) * 2 - 1;
	/////////////////////////////////////////////////////////////////////////////

	//////////////////// Test model from mean of images /////////////////////////
	Mat alt_pred = (val_data * _pos_sumF.t() > 0) / 255;
	double alt_true = alt_pred.size().height - sum((alt_pred == val_gnd) / 255)[0];
	double alt_pct = (alt_true / (double) alt_pred.size().height) * 100.0;
	cout << "Validation correct with mean model: " << alt_pct << "%" << endl;
	/////////////////////////////////////////////////////////////////////////////

	/////////////////////////////// Train SVM ///////////////////////////////////
	cout << endl << "Build SVM model" << endl;

	Mat best_W;
	double best_b = 0;
	double best_c = -DBL_MAX;
	double best_pct = -DBL_MAX;

	MySVM svm;
	const double C = Detector::cfg()->getValue<double>("settings/svm/params/C");
	cout << endl << "line:" << __LINE__ << ") C: " << C << endl;

	// Set up SVM's parameters
	SVMParams params;
	params.svm_type = SVM::C_SVC;
	params.C = C;
	params.kernel_type = SVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, _max_count, _epsilon);

	Mat data;
	if (train_data.type() != CV_32F)
		train_data.convertTo(data, CV_32F);
	else
		data = train_data;

	// Train the SVM
	cout << "line:" << __LINE__ << ") Training SVM..." << endl;
	svm.train(data, labels, Mat(), Mat(), params);

	Mat labels_train;
	svm.predict(data, labels_train);

	Mat labels_32F;
	labels.convertTo(labels_32F, CV_32F);
	cv::Mat diff = labels_32F == labels_train;
	double train_true = countNonZero(diff);
	double train_pct = (train_true / (double) diff.rows) * 100.0;
	cout << "\tTraining correct: " << train_pct << "%" << endl;

	const int sv_count = svm.get_support_vector_count();
	const int sv_length = svm.get_var_count();
	cout << "\tSupport vector(s): " << sv_count << ", vector-length: " << sv_length << endl;

	CvSVMDecisionFunc* decision = svm.getDecisionFunc();
	Mat W = Mat::zeros(1, sv_length, CV_64F);
	for (int i = 0; i < sv_count; ++i)
	{
		// Compute W from support_vector and decision->alpha
		const float* support_vector = svm.get_support_vector(i);
		for (int j = 0; j < sv_length; ++j)
			W.at<double>(0, j) += decision->alpha[i] * support_vector[j];
	}

	const double b = -decision->rho;
	cout << "line:" << __LINE__ << ") bias: " << b << endl;

	/*
	 {
	 // Compute the confidence values for training and validation as the distances
	 // between the sample vectors X and weight vector W, using bias b:
	 // conf = X * W + b
	 //
	 // Approach this as a matrix calculation (ie. fill in the dots below, using no
	 // more than a single line for calculating respectively conf_train and conf_val)
	 //
	 // The confidence value for training should be the same value you get from
	 // svm.predict(data, labels_train);

	 Mat conf_train = ...;
	 Mat conf_val = ...;
	 Mat train_pred = (conf_train > 0) / 255;
	 Mat val_pred = (conf_val > 0) / 255;
	 double train_true = train_pred.rows - sum((train_pred == train_gnd) / 255)[0];
	 double train_pct = (train_true / (double) train_pred.rows) * 100.0;
	 double val_true = val_pred.rows - sum((val_pred == val_gnd) / 255)[0];
	 double val_pct = (val_true / (double) val_pred.rows) * 100.0;
	 cout << "\tTraining correct: " << train_pct << "%" << endl;
	 cout << "\tValidation correct: " << val_pct << "%" << endl;
	 }
	 */

	best_W = W;
	best_b = b;
	best_c = C;

	Mat W_rect(height, width, CV_64F);
	W_rect.data = best_W.data;
	assert((int ) best_W.total() == width * height);

	normalize(W_rect, W_rect, 255, 0, NORM_MINMAX);
	Mat W_img, nW_img;
	W_rect.convertTo(W_img, CV_8U);
	bitwise_not(W_img, nW_img);
	imshow("Model", nW_img);
	cout << "Press a key to continue" << endl;
	waitKey();
	/////////////////////////////////////////////////////////////////////////////

	////////////////////////////// Test on real image ///////////////////////////
	cout << "Detecting faces..." << endl;

	// Read ground truth if we consider data/img1.jpg
	vector<Rect> ground_truths;
	if (_query_image_file == Detector::cfg()->getValue<string>("settings/images/test@file"))
	{
		QXmlElms faces_xml = Detector::cfg()->getValues("settings/images/test/ground_truth/face");
		for (size_t r = 0; r < faces_xml.size(); ++r)
		{
			QXmlElmPtr face_xml = faces_xml[r];
			int id = Detector::cfg()->getValue<int>("@id", face_xml);
			int width = Detector::cfg()->getValue<int>("width", face_xml);
			int height = Detector::cfg()->getValue<int>("height", face_xml);
			int x = Detector::cfg()->getValue<int>("x", face_xml);
			int y = Detector::cfg()->getValue<int>("y", face_xml);
			ground_truths.push_back(Rect(x, y, width, height));
		}
	}

	Mat query_image = imread(_query_image_file);
	double bside = MAX(query_image.rows, query_image.cols);
	int t_height = query_image.rows * (MIN(_max_image_size, bside) / bside);
	int t_width = query_image.cols * (MIN(_max_image_size, bside) / bside);
	resize(query_image, query_image, Size(t_width, t_height));
	Mat query_image_gr;
	cvtColor(query_image, query_image_gr, CV_BGR2GRAY);

	// Create feature pyramid
	vector<Mat*> pyramid;
	createPyramid(query_image_gr, pyramid);

	map<int, vector<int>> Xvs, Yvs;							// Bounding box locations per layer
	Mat detections;
	vector<Range> layers;

	cout << "Processing " << pyramid.size() << " pyramid layers" << endl;
	vector<double> fps;
	int64 t0 = Utility::get_time_curr_tick();

	for (size_t layer = 0; layer < pyramid.size(); ++layer)
	{
		string etf = Utility::show_fancy_etf(layer, pyramid.size(), 1, t0, fps);
		if (!etf.empty()) cout << etf << endl;

		// Generate subwindow locations
		Range rx(0, pyramid.at(layer)->cols - _model_size.width);
		Range ry(0, pyramid.at(layer)->rows - _model_size.height);
		Mat1i X, Y;
		Utility::meshgrid(rx, ry, X, Y);
		Mat X1 = X.reshape(1, 1);
		Mat Y1 = Y.reshape(1, 1);
		const int* px = X1.ptr<int>(0);
		const int* py = Y1.ptr<int>(0);
		vector<int> Xv(px, px + X1.total()), Yv(py, py + Y1.total());
		Xvs.insert(make_pair(layer, Xv));
		Yvs.insert(make_pair(layer, Yv));

		// Extract subwindows from image for detection
		Mat sub_windows;
		Mat mv_sub_data, sv_sub_data;
		for (size_t i = 0; i < X1.total(); ++i)
		{
			Mat sub = (*pyramid.at(layer))(Rect(Point(Xv[i], Yv[i]), _model_size)).clone();
			Mat sub1d = sub.reshape(1, 1);
			Mat sub1dF;
			sub1d.convertTo(sub1dF, CV_64F);

			Mat mean, stddev;
			meanStdDev(sub1dF, mean, stddev);
			Mat mean_line(1, sub1dF.cols, CV_64F);
			mean_line = mean.at<double>(0, 0);
			Mat std_line(1, sub1dF.cols, CV_64F);
			std_line = stddev.at<double>(0, 0);
			mv_sub_data.push_back(mean_line);
			sv_sub_data.push_back(std_line);
			sub_windows.push_back(sub1dF);
		}

		if (_do_equalizing)
		{
			// Equalize image (subtract mean, divide by stddev)
			sub_windows = sub_windows - mv_sub_data;
			sub_windows = sub_windows / sv_sub_data;
		}

		if (_do_whitening)
		{
			Utility::whiten(sub_windows, _model_size, sub_windows);
		}

		/*
		 * If you have found the answer to the question above,
		 * you can replace the loop below by a single line
		 *
		 * Mat detect = ...;
		 */
		Mat detect(sub_windows.rows, 1, CV_64F);
		for (int i = 0; i < sub_windows.rows; ++i)
		{
			Mat sub1dF;
			if (sub_windows.type() == CV_32F)
				sub1dF = sub_windows.row(i);
			else
				sub_windows.row(i).convertTo(sub1dF, CV_32F);

			detect.at<double>(i, 0) = (double) svm.predict(sub1dF, true);
		}

		// Show detection results as a heatmap of most likely face locations for this pyramid layer
		Mat face_locations = Mat(detect.t()).reshape(detect.channels(),
				(pyramid.at(layer)->size().height - _model_size.height) + 1);
		Mat heatmap;
		Utility::get_heatmap(-face_locations, heatmap);
		namedWindow("Face heatmap", CV_WINDOW_KEEPRATIO);
		imshow("Face heatmap", heatmap);
		waitKey(200);

		layers.push_back(Range(detections.rows, detections.rows + detect.rows));
		detections.push_back(detect);
	}

	// Delete the pyramid layer images
	for (size_t p = 0; p < pyramid.size(); ++p)
		delete pyramid[p];

	//Normalize the results
	Mat n_detect;
	const int min_value = 50;
	const int max_value = 100;
	normalize(detections, n_detect, max_value, 0, NORM_MINMAX);

	//Create window
	namedWindow("Search image", CV_WINDOW_KEEPRATIO);

	//Trackbar with default values for detection threshold
	int value = _initial_threshold - min_value, o_val = -INT_MAX;
	createTrackbar("Threshold", "Search image", &value, max_value - min_value);

	//Holder for detection bounding boxes
	vector<Rect> rects;
	vector<double> scores;

	//Drawing loop
	int key = -1;
	while (key == -1)
	{
		int o_value = min_value + value;
		if (o_value != o_val)
		{
			double threshold = ((max_value - o_value) / 100.0) * max_value;

			//Create result vector with all detections above the threshold
			vector<ResultLocation> results;
			for (int i = 0; i < n_detect.rows; ++i)
			{
				if (n_detect.at<double>(i, 0) < threshold)
				{
					//Find correct layer for this detection
					int offset = 0;
					int layer_n = 0;
					for (size_t l = 0; l < layers.size(); ++l)
					{
						Range layer = layers.at(l);
						if (i >= layer.start && i < layer.end)
						{
							layer_n = l;
							break;
						}
						offset += layer.size();
					}

					ResultLocation result;
					result.layer = layer_n;
					result.offset = i - offset;
					result.score = n_detect.at<double>(i, 0);
					results.push_back(result);
				}
			}

			// Sort the detection results high to low
			sort(results.begin(), results.end(), descending());

			Mat bboxs;
			rects.clear();
			scores.clear();
			for (size_t r = 0; r < results.size(); ++r)
			{
				ResultLocation result = results[r];
				//Look up correct bounding box location and scale to correct size
				Rect rect(Point(Xvs[result.layer][result.offset], Yvs[result.layer][result.offset]), _model_size);
				double scale = pow(_layer_scale_factor, (double) result.layer);
				rect.x = rect.x * scale;
				rect.y = rect.y * scale;
				rect.width = rect.height * scale;
				rect.height = rect.height * scale;

				// Non-Maximum Suppression of found bounding boxes
				if (!bboxs.empty())
				{
					double m_overl;
					minMaxLoc(Utility::box_overlap(bboxs, rect), NULL, &m_overl);
					if (m_overl < _overlap_threshold)
					{
						rects.push_back(rect);
						scores.push_back(result.score);
					}
					else
						continue;
				}
				else
				{
					rects.push_back(rect);
					scores.push_back(result.score);
				}

				Mat bbox = (Mat_<double>(1, 4) << rect.tl().x, rect.tl().y, rect.br().x, rect.br().y);
				bboxs.push_back(bbox);
			}

			/*
			 * Iterate over all detected candidates and compare to the ground truth data (if any)
			 */
			if (!ground_truths.empty())
			{
				vector<Rect> search_gt = ground_truths;
				for (size_t r = 0; r < rects.size(); ++r)
				{
					Rect detection = rects.at(r);

					vector<Rect>::iterator it = search_gt.begin();
					for (; it != search_gt.end();)
					{
						Rect ground_truth = *it;
						double intersection_size = 0;

						if (intersection_size >= _gt_accuracy)
						{
							// if size is equal or larger than the accuracy threshold we have found a true positive
							it = search_gt.erase(it);
						}
						else
						{
							++it;
						}
					}
				}

				// Calculate the Precision and Recall at the currect threshold value
				double precision = 0, recall = 0;
				char p_buf[32], r_buf[32];
				sprintf(p_buf, "%4.2f", precision);
				sprintf(r_buf, "%4.2f", recall);
				cout << "Precision: " << p_buf << "%\tRecall: " << r_buf << "%" << endl;
			}

			o_val = o_value;
		}

		// Convert image to 3 channels, just so we can place bounding boxes in red
		Mat image;
		if (query_image.channels() == 1)
			cvtColor(query_image, image, CV_GRAY2BGR);
		else
			image = query_image.clone();

		for (size_t r = 0; r < rects.size(); ++r)
		{
			Rect rect = rects.at(r);
			double score = scores.at(r);
			rectangle(image, rect, Color_RED);

			stringstream text;
			text << (1 - score / max_value);
			double scale = 0.8;
			int font = CV_FONT_HERSHEY_PLAIN;
			int thickness = 1;
			int baseline = 0;
			cv::Size size = cv::getTextSize(text.str(), font, scale, thickness, &baseline);
			rectangle(image, rect, Color_RED, 2);
			putText(image, text.str(), Point(rect.x, rect.y + size.height), font, scale, Color_YELLOW, thickness, CV_AA);
		}

		imshow("Search image", image);
		key = waitKey(50);
	}
	/////////////////////////////////////////////////////////////////////////////
}

} /* namespace nl_uu_science_gmt */

int main(int argc, char** argv)
{
	std::string query_image_file = Detector::cfg()->getValue<string>("settings/images/test@file");
	if (argc > 1) query_image_file = (std::string) argv[1];
	cout << "Testing on: " << query_image_file << endl;

	Detector detector(query_image_file);
	detector.run();
}
