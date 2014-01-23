/*
 * Detector.h
 *
 *  Created on: Aug 23, 2013
 *      Author: coert
 */

#ifndef DETECTOR_H_
#define DETECTOR_H_

#include <memory>
#include <opencv2/opencv.hpp>

#include "Utility.h"
#include "FileIO.h"
#include "QueryXML.h"
#include "MySVM.h"

namespace nl_uu_science_gmt
{

class Detector
{
	static QueryXML* _Config;
	static std::string ConfigPath;

	const int _pos_amount;
	const int _target_width;
	const int _posneg_factor; // 4x as many negative images as positive ones
	const int _seed;
	const int _disp;
	const size_t _max_count;
	const double _epsilon;
	const int _max_image_size;
	const int _layer_scale_interval;
	const int _initial_threshold;
	const double _overlap_threshold;
	const bool _do_equalizing;
	const bool _do_whitening;
	const double _gt_accuracy;
	const std::string _query_image_file;

	cv::Size _model_size;
	cv::Mat _neg_sum8U, _pos_sum8U, _pos_sumF;
	double _layer_scale_factor;

	void readPosFilelist(std::vector<std::string> &);
	void readNegFilelist(std::vector<std::string> &);

	void readPosData(const std::vector<std::string> &, cv::Mat &);
	void readNegData(const std::vector<std::string> &, cv::Mat &);

	void createPyramid(const cv::Mat &, std::vector<cv::Mat*> &);

public:
	Detector(const std::string &);
	virtual ~Detector();

	void run();

	static std::string StorageExt;
	static std::string ImageExt;

	static QueryXML*
	cfg()
	{
		if (!_Config) // Only allow one instance of class to be generated.
			_Config = new QueryXML(ConfigPath);

		return _Config;
	}

	static void cfg_destroy()
	{
		if (_Config) delete _Config;
		_Config = NULL;
	}
};

} /* namespace nl_uu_science_gmt */
#endif /* DETECTOR_H_ */
