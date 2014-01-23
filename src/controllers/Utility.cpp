/*
 * Utility.cpp
 *
 *  Created on: Jan 9, 2014
 *      Author: coert
 */

#include "Utility.h"

namespace nl_uu_science_gmt
{

/**
 * Compute the symmetric intersection over union overlap between a set of
 * bounding boxes in a and a single bounding box in b
 *
 * @param a A matrix in which each row specifies a bounding box
 * @param b A single bounding box
 * @returns A matrix containing the symmetric intersection over union overlap between a and b
 */
cv::Mat Utility::box_overlap(const cv::Mat &a, const cv::Rect &b)
{
	cv::Mat x1(a.rows, 1, CV_64F);
	cv::Mat y1(a.rows, 1, CV_64F);
	cv::Mat x2(a.rows, 1, CV_64F);
	cv::Mat y2(a.rows, 1, CV_64F);

	for (int r = 0; r < a.rows; ++r)
	{
		x1.at<double>(r, 0) = MAX(a.at<double>(r, 0), b.tl().x);
		y1.at<double>(r, 0) = MAX(a.at<double>(r, 1), b.tl().y);
		x2.at<double>(r, 0) = MIN(a.at<double>(r, 2), b.br().x);
		y2.at<double>(r, 0) = MIN(a.at<double>(r, 3), b.br().y);
	}

	cv::Mat w = x2 - x1 + 1;
	cv::Mat h = y2 - y1 + 1;
	cv::Mat inter;
	cv::multiply(w, h, inter);

	cv::Mat a_area(a.rows, 1, CV_64F);
	for (int r = 0; r < a.rows; ++r)
	{
		int width = a.at<double>(r, 2) - a.at<double>(r, 0) + 1;
		int height = a.at<double>(r, 3) - a.at<double>(r, 1) + 1;
		a_area.at<double>(r, 0) = width * height;
	}

	cv::Mat o;
	cv::divide(inter, a_area + b.area() - inter, o);
	assert(o.rows > 0);

	std::vector<cv::Point> labels_w, labels_h;
	mat_labels_lteq(w, 0, labels_w);
	for (int l = 0; l < labels_w.size(); ++l)
		o.at<double>(labels_w[l]) = 0;
	mat_labels_lteq(h, 0, labels_h);
	for (int h = 0; h < labels_h.size(); ++h)
		o.at<double>(labels_h[h]) = 0;

	return o;
}

/**
 * Returns a date string
 */
#ifdef _WIN32
std::string Utility::get_datestamp(__time64_t unix_t)
{
	__time64_t in_time;
#elif __linux__
std::string Utility::get_datestamp(time_t unix_t)
{
	time_t in_time;
#endif

	if (unix_t == 0)
	{
#ifdef _WIN32
		in_time = _time64(NULL);
#elif __linux__
		in_time = time(NULL);
#endif
	}
	else
	{
		in_time = unix_t;
	}

#ifndef _WIN32
	tm tm_buf;
	struct tm* t = localtime_r(&in_time, &tm_buf);
#else
	struct tm *t;
	__time64_t long_time;
	_time64(&long_time);
	t = _localtime64( &long_time );
#endif

	std::stringstream date_day;
	date_day << t->tm_mday;
	std::string d = date_day.str();
	if (d.length() < 2) d = "0" + d;

	std::stringstream date_m;
	date_m << (1 + t->tm_mon);
	std::string m = date_m.str();
	if (m.length() < 2) m = "0" + m;

	std::stringstream date_y;
	date_y << (1900 + t->tm_year);
	std::string y = date_y.str();

	std::stringstream date_string;
	date_string << d << "-" << m << "-" << y;

	return date_string.str();
}

/**
 * Returns a time string
 */
#ifdef _WIN32
std::string Utility::get_timestamp(__time64_t unix_t)
{
	__time64_t in_time;
#elif __linux__
std::string Utility::get_timestamp(time_t unix_t)
{
	time_t in_time;
#endif

	if (unix_t == 0)
	{
#ifdef _WIN32
		in_time = _time64(NULL);
#elif __linux__
		in_time = time(NULL);
#endif
	}
	else
	{
		in_time = unix_t;
	}

#ifndef _WIN32
	tm tm_buf;
	struct tm* t = localtime_r(&in_time, &tm_buf);
#else
	struct tm *t;
	__time64_t long_time;
	_time64(&long_time);
	t = _localtime64( &long_time );
#endif

	std::stringstream time_hr;
	time_hr << t->tm_hour;
	std::string hr = time_hr.str();
	if (hr.length() < 2) hr = "0" + hr;

	std::stringstream time_mnt;
	time_mnt << t->tm_min;
	std::string mnt = time_mnt.str();
	if (mnt.length() < 2) mnt = "0" + mnt;

	std::stringstream time_sec;
	time_sec << t->tm_sec;
	std::string sec = time_sec.str();
	if (sec.length() < 2) sec = "0" + sec;

	std::stringstream date_string;
	date_string << get_datestamp() << " " << hr << ":" << mnt << ":" << sec;

	return date_string.str();
}

/**
 * Calculate progress and estimates time when finished
 */
std::string Utility::get_time_progress(const int index, const int total, std::vector<double> &t_list)
{
	int progress = index;
	int left = total - progress;
	double pct = (progress / (double) total) * 100.f;
	char pct_buf[32];
	sprintf(pct_buf, "%4.2f", pct);
	double mfps = median(t_list);
	char mfps_buf[32];
	sprintf(mfps_buf, "%4.2f", mfps);
	std::string eta = "n/a";
	double fps = 0;

	if (mfps > 0)
	{
		double time_left = left / mfps;
		if (time_left < std::numeric_limits<double>::max()) eta = get_timestamp(time(NULL) + (time_t) round(time_left));
		fps = t_list.size() > 0 ? t_list.back() : -1;
	}

	char fps_buf[32];
	sprintf(fps_buf, "%4.2f", fps);
	std::stringstream r_string;
	r_string << pct_buf << "% (" << progress << "/" << total << ")";
	//r_string << ", fps: " << fps_buf << " (mfps: " << mfps_buf << ", etf: " << eta << ")";
	return r_string.str();
}

/**
 * Wrapper for showing progress
 */
std::string Utility::show_fancy_etf(const long index, const long total, const int stride, int64 &t0,
		std::vector<double> &fps)
{
	std::string line;
	if (stride == 0 || index % stride == 0)
	{
		int64 t1 = get_time_curr_tick();
		double past = get_time_past_sec(t0, t1);
		fps.push_back(stride / past);
		line = get_time_progress(index + 1, total, fps);
		t0 = t1;
	}

	return line;
}

/**
 * Matlab Rectangular Grid
 * See http://www.mathworks.nl/help/matlab/ref/meshgrid.html
 */
void Utility::meshgrid(const std::vector<std::vector<double>> &r, std::vector<cv::Mat*> &M)
{
	assert(r.size() == 2 || r.size() == 3);

	std::vector<int> m_size;
	for (size_t i = 0; i < r.size(); ++i)
		m_size.push_back(r[i].size());

	for (size_t i = 0; i < r.size(); ++i)
	{
		cv::Mat m(r.size(), &m_size[0], CV_64F, cv::Scalar::all(0));

		for (int d = 0; d < m_size[0]; ++d)
		{
			cv::Mat filled;
			if (m.dims == 3 && i == 0)
			{
				cv::Mat sub2d(1, 1, CV_64F, cv::Scalar::all(r[0][d]));
				cv::repeat(sub2d, m.size[1], m.size[2], filled);
			}
			else if ((m.dims == 3 && i == 1) || (m.dims == 2 && i == 0))
			{
				cv::Mat sub2d(r[m.dims == 3 ? 1 : 0]);
				cv::repeat(sub2d, 1, m.size[m.dims == 3 ? 2 : 1], filled);
			}
			else if ((m.dims == 3 && i == 2) || (m.dims == 2 && i == 1))
			{
				cv::Mat sub2d(r[m.dims == 3 ? 2 : 1]);
				cv::repeat(sub2d.t(), m.size[m.dims == 3 ? 1 : 0], 1, filled);
			}

			if (m.dims == 3)
			{
				cv::Range range[] = { cv::Range(d, d + 1), cv::Range::all(), cv::Range::all() };
				std::vector<int> sizen(3);
				sizen[0] = 1;
				sizen[1] = filled.size().height;
				sizen[2] = filled.size().width;
				cv::Mat n(m.dims, &sizen[0], CV_64F);
				n.data = filled.data;
				cv::Range rangen[] = { cv::Range(0, 1), cv::Range::all(), cv::Range::all() };
				n(rangen).copyTo(m(range));
			}
			else if (m.dims == 2)
			{
				filled.copyTo(m);
			}
		}

		M.push_back(new cv::Mat(m));
	}
}

/**
 * Shift zero-frequency component to center of spectrum
 * See: http://www.mathworks.nl/help/matlab/ref/fftshift.html
 */
cv::Mat Utility::fftshift(const cv::Mat &src, int dim)
{
	std::vector<std::vector<int>> idx;
	if (dim != 0)
	{
		//TODO
		assert(false);
	}
	else
	{
		for (int d = 0; d < src.dims; ++d)
		{
			int m = src.size[d];
			int p = std::ceil(m / 2.0);
			UniqueValue_i uv1(p, m - 1);
			std::vector<int> row_front(uv1._size);
			std::generate(row_front.begin(), row_front.end(), uv1);
			UniqueValue_i uv2(0, p - 1);
			std::vector<int> row_end(uv2._size);
			std::generate(row_end.begin(), row_end.end(), uv2);
			row_front.insert(row_front.end(), row_end.begin(), row_end.end());
			idx.emplace_back(row_front);
		}
	}

	cv::Mat s0 = src;
	for (size_t i = 0; i < idx.size(); ++i)
	{
		cv::Mat s1 = cv::Mat(s0.dims, s0.size, s0.type());
		for (size_t j = 0; j < idx[i].size(); ++j)
		{
			std::vector<cv::Range> range0(idx.size());
			std::vector<cv::Range> range1(idx.size());
			for (size_t r = 0; r < idx.size(); ++r)
			{
				if (r == i)
				{
					range0[i] = cv::Range(idx[i][j], idx[i][j] + 1);
					range1[i] = cv::Range(j, j + 1);
				}
				else
				{
					range0[r] = cv::Range::all();
					range1[r] = cv::Range::all();
				}
			}
			cv::Mat d = s0(&range0[0]);
			cv::Mat x0 = s1(&range1[0]);
			d.copyTo(x0);
		}
		s0 = s1.clone();
	}
	return s0;
}

/**
 * Matrix whitening
 * See: http://en.wikipedia.org/wiki/Whitening_transformation
 */
void Utility::whiten(const cv::Mat &src, const cv::Size &s_size, cv::Mat &dst)
{
	std::vector<int> size(2);
	size[0] = s_size.width;
	size[1] = s_size.height;

	assert(src.dims == 2 && src.channels() == 1 && (src.type() == CV_8U || src.type() == CV_32F || src.type() == CV_64F));

	cv::Mat f_images;
	if (src.type() != CV_32F)
		src.convertTo(f_images, CV_32F);
	else
		f_images = src;

	std::vector<std::vector<double>> ranges;
	ranges.reserve(size.size());
	for (size_t s = 0; s < size.size(); ++s)
	{
		int n = size[s];
		double rmin = -n / 2.0;
		double rmax = n / 2.0;
		UniqueValue_d uv(rmin, rmax);
		std::vector<double> range(uv._size);
		std::generate(range.begin(), range.end(), uv);
		ranges.emplace_back(range);
	}

	std::vector<cv::Mat*> M;
	meshgrid(ranges, M);

	cv::Mat f_sum;
	for (size_t s = 0; s < size.size(); ++s)
	{
		cv::Mat f = *M[s];
		cv::Mat f_32s;
		f.convertTo(f_32s, CV_32F);
		cv::Mat fs;
		cv::multiply(f_32s, f_32s, fs);

		if (s == 0)
			f_sum = fs;
		else
			cv::add(f_sum, fs, f_sum);
	}

	for (size_t m = 0; m < M.size(); ++m)
		delete M[m];

	cv::Mat rho;
	cv::sqrt(f_sum, rho);

	int size_sum = std::accumulate(size.begin(), size.end(), 0);
	double f_0 = 0.4 * (size_sum / (double) size.size());

	cv::Mat filt_f_0, filt_pow, filt_exp, filt;
	cv::divide(rho, f_0, filt_f_0);
	cv::pow(filt_f_0, 4, filt_pow);
	cv::exp(-filt_pow, filt_exp);
	cv::multiply(rho, filt_exp, filt);
	cv::Mat filt_shifted = fftshift(filt);
	cv::Mat filt_shifted_1D(1, filt_shifted.total(), filt_shifted.type());
	filt_shifted_1D.data = filt_shifted.data;

	dst = cv::Mat(src.size(), src.type());

	for (int i = 0; i < f_images.rows; ++i)
	{
		cv::Mat frame = f_images.row(i);

		cv::Mat fourierTransform;
		cv::dft(frame, fourierTransform, cv::DFT_COMPLEX_OUTPUT);
		std::vector<cv::Mat> complex_parts;
		cv::split(fourierTransform, complex_parts);

		cv::Mat real;
		cv::multiply(complex_parts[0], filt_shifted_1D, real);
		cv::Mat imagin;
		cv::multiply(complex_parts[1], filt_shifted_1D, imagin);

		complex_parts[0] = real;
		complex_parts[1] = imagin;
		cv::Mat complex;
		cv::merge(complex_parts, complex);

		cv::Mat inverseTransform;
		cv::dft(complex, inverseTransform, cv::DFT_REAL_OUTPUT | cv::DFT_INVERSE | cv::DFT_SCALE);

		inverseTransform.reshape(1, 1).copyTo(dst.row(i));
	}

	cv::Mat var(dst.rows, 1, dst.type());
	for (int i = 0; i < dst.rows; ++i)
	{
		cv::Mat mean, stddev;
		cv::meanStdDev(dst.row(i), mean, stddev);
		cv::Mat row_var;
		cv::pow(stddev, 2, row_var);
		row_var.copyTo(var.row(i));
	}

	cv::Mat root;
	sqrt(cv::mean(var), root);
	cv::Mat div_dst;
	cv::divide(dst, root, div_dst);
	cv::Mat mult_dst;
	cv::multiply(sqrt(0.1), div_dst, mult_dst);

	cv::Mat whitened(src.size(), src.type());
	for (int i = 0; i < mult_dst.rows; ++i)
		mult_dst.row(i).copyTo(whitened.row(i));

	if (whitened.type() != src.type())
		whitened.convertTo(dst, src.type());
	else
		dst = whitened;
}

void Utility::whiten(const cv::Mat &src, const std::vector<int> &size, cv::Mat &dst)
{
	whiten(src, cv::Size(size[0], size[1]), dst);
}

void Utility::get_images_canvas(const int &c_size, const cv::Mat &data, const cv::Size &size, cv::Mat &canvas)
{
	get_images_canvas(cv::Size(c_size, c_size), data, size, canvas);
}

/**
 * Creates a canvas of multiple images (collage)
 */
void Utility::get_images_canvas(const cv::Size &c_size, const cv::Mat &data, const cv::Size &size, cv::Mat &canvas)
{
	const int canvas_total = MIN(c_size.area(), data.rows);
	const int canvas_cols = c_size.width;

	cv::Mat tmp_sz_im_line;
	for (int i = 0; i < canvas_total; ++i)
	{
		cv::Mat tmp_im = data.row(i).reshape(data.channels(), size.height);
		cv::normalize(tmp_im, tmp_im, 255, 0, cv::NORM_MINMAX);

		cv::Mat tmp_sz_im;
		if (tmp_im.channels() == 1)
			tmp_im.convertTo(tmp_sz_im, CV_8U);
		else if (tmp_im.channels() == 3) tmp_im.convertTo(tmp_sz_im, CV_8UC3);

		if (i % canvas_cols == 0)
		{
			if (canvas.empty())
				canvas = tmp_sz_im_line;
			else
				vconcat(canvas, tmp_sz_im_line, canvas);

			tmp_sz_im_line = tmp_sz_im;
		}
		else
		{
			hconcat(tmp_sz_im_line, tmp_sz_im, tmp_sz_im_line);
		}
	}
	vconcat(canvas, tmp_sz_im_line, canvas);
}

/**
 * Creates a Heatmap
 *
 * int color_map:
 * 		COLORMAP_AUTUMN, COLORMAP_BONE, COLORMAP_JET, COLORMAP_WINTER, COLORMAP_RAINBOW, COLORMAP_OCEAN,
 *		COLORMAP_SUMMER, COLORMAP_SPRING, COLORMAP_COOL, COLORMAP_HSV, COLORMAP_PINK, COLORMAP_HOT
 */
void Utility::get_heatmap(const cv::Mat &src, cv::Mat &dst, int size, int offset, int color_map, bool normalize)
{
	assert(src.channels() == 1 && src.dims == 2);

	int target_width = 100;
	cv::Mat heat;
	if (src.cols > target_width)
	{
		int height = round(src.rows * (target_width / (double) src.cols));
		cv::resize(src, heat, cv::Size(target_width, height));
	}
	else
	{
		heat = src;
	}
	cv::Mat cl_filter = heat.clone() + offset;

	cv::Mat img;
	if (normalize)
		cv::normalize(cl_filter, img, 255, 0, cv::NORM_MINMAX);
	else
		img = cl_filter;
	cv::Mat proc_img;
	img.convertTo(proc_img, CV_32F);

	cv::Mat heatmap = cv::Mat::zeros(cv::Size(proc_img.cols * size, proc_img.rows * size), CV_8UC3);
	for (int y = 0; y < proc_img.rows; ++y)
	{
		for (int x = 0; x < proc_img.cols; ++x)
		{
			cv::rectangle(heatmap, cv::Point(x * size, y * size), cv::Point(x * size + size, y * size + size),
					cv::Scalar::all(round(proc_img.at<float>(y, x))), CV_FILLED);
		}
	}

	cv::applyColorMap(heatmap, dst, color_map);
}

} /* namespace nl_uu_science_gmt */
