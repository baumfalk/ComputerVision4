/*
 * Utility.h
 *
 *  Created on: Jan 9, 2014
 *      Author: coert
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <math.h>
#include <memory>
#include <numeric>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <time.h>

#include <opencv2/opencv.hpp>

namespace nl_uu_science_gmt
{

const static cv::Scalar Color_BLUE = cv::Scalar(255, 0, 0);
const static cv::Scalar Color_GREEN = cv::Scalar(0, 200, 0);
const static cv::Scalar Color_RED = cv::Scalar(0, 0, 255);
const static cv::Scalar Color_YELLOW = cv::Scalar(0, 255, 255);
const static cv::Scalar Color_MAGENTA = cv::Scalar(255, 0, 255);
const static cv::Scalar Color_CYAN = cv::Scalar(255, 255, 0);
const static cv::Scalar Color_WHITE = cv::Scalar(255, 255, 255);
const static cv::Scalar Color_BLACK = cv::Scalar(0, 0, 0);

class Utility
{
	struct UniqueValue_i
	{
		int _current;
		int _max;
		int _step;
		int _size;

		UniqueValue_i(const int current = 0, const int max = 1, const int step = 1) :
				_current(current), _max(max), _step(step)
		{
			_size = 1 + ((_max - _current) / _step);
		}

		double operator()()
		{
			int val = _current;
			_current += _step;
			return val;
		}
	};

	struct UniqueValue_d
	{
		double _current;
		double _max;
		double _step;
		int _size;

		UniqueValue_d(const double current = 0, const double max = 1, const double step = 1) :
				_current(current), _max(max), _step(step)
		{
			_size = floor((_max - _current) / _step);
		}

		double operator()()
		{
			double val = _current;
			_current += _step;
			return val;
		}
	};

	static double round(double value, int digits = 0)
	{
		return floor(value * pow((double) 10, (double) digits) + 0.5) / pow((double) 10, (double) digits);
	}

public:

	/**
	 * Vector median
	 */
	template<typename T>
	static inline T median(std::vector<T> values)
	{
		size_t n = values.size() / 2;
		if (n > 0)
		{
			std::nth_element(values.begin(), values.begin() + n, values.end());
			return values.at(n);
		}
		else
		{
			return 0;
		}
	}

	template<typename T>
	static void mat_labels_lteq(const cv::Mat_<T> &src, double boundary_value, std::vector<cv::Point> &labels)
	{
		cv::Mat t_m = src;

		for (int i = 0; i < src.cols; i++)
		{
			for (int j = 0; j < src.rows; j++)
			{
				cv::Point point(i, j);
				T value = src(point);
				if (mat_labels_lteq(value, boundary_value)) labels.push_back(point);
			}
		}
	}

	template<typename T, int c>
	static bool mat_labels_lteq(cv::Vec<T, c> vector, double boundary_value)
	{
		T* val = vector.val;
		for (int i = 0; i < c; i++)
			if (val[i] <= boundary_value) return true;

		return false;
	}

	template<typename T>
	static bool mat_labels_lteq(T value, double boundary_value)
	{
		return value <= boundary_value;
	}

	static inline void mat_labels_lteq(const cv::Mat &src, double boundary_value, std::vector<cv::Point> &labels)
	{
		switch (src.type())
		{
		case CV_8U:
		{
			cv::Mat_<uchar> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_8UC2:
		{
			cv::Mat_<cv::Vec2b> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_8UC3:
		{
			cv::Mat_<cv::Vec3b> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_32S:
		{
			cv::Mat_<double> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_32F:
		{
			cv::Mat_<double> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_32FC2:
		{
			cv::Mat_<cv::Vec2f> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_32FC3:
		{
			cv::Mat_<cv::Vec3f> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_64F:
		{
			cv::Mat_<double> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_64FC2:
		{
			cv::Mat_<cv::Vec2d> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		case CV_64FC3:
		{
			cv::Mat_<cv::Vec3d> type = src;
			mat_labels_lteq(type, boundary_value, labels);
			break;
		}
		}
	}

	static cv::Mat box_overlap(const cv::Mat &, const cv::Rect &);

#ifdef _WIN32
	static std::string get_datestamp(__time64_t unix_t = 0);
	static std::string get_timestamp(__time64_t unix_t = 0);
#elif __linux__
	static std::string get_datestamp(time_t unix_t = 0);
	static std::string get_timestamp(time_t unix_t = 0);
#endif

	static inline int64 get_time_curr_tick()
	{
		return cv::getTickCount();
	}

	static inline double get_time_past_sec(const int64 t0, const int64 t1)
	{
		return ((double(t1 - t0) / cvGetTickFrequency()) / 1e+6);
	}

	static std::string get_time_progress(const int, const int, std::vector<double> &);

	static std::string show_fancy_etf(const long, const long, const int, int64 &, std::vector<double> &);

	/**
	 * Matlab Rectangular Grid
	 * See http://www.mathworks.nl/help/matlab/ref/meshgrid.html
	 */
	template<typename T>
	static void meshgrid(const cv::Range &rx, const cv::Range &ry, T &X, T &Y)
	{
		switch (X.type())
		{
		case CV_32S:
		{
			std::vector<int> t_x, t_y;
			for (int i = rx.start; i <= rx.end; i++)
				t_x.push_back(i);
			for (int i = ry.start; i <= ry.end; i++)
				t_y.push_back(i);
			meshgrid(t_x, t_y, X, Y);
		}
			break;
		case CV_32F:
		{
			std::vector<float> t_x, t_y;
			for (int i = rx.start; i <= rx.end; i++)
				t_x.push_back(i);
			for (int i = ry.start; i <= ry.end; i++)
				t_y.push_back(i);
			meshgrid(t_x, t_y, X, Y);
		}
			break;
		case CV_64F:
		{
			std::vector<double> t_x, t_y;
			for (int i = rx.start; i <= rx.end; i++)
				t_x.push_back(i);
			for (int i = ry.start; i <= ry.end; i++)
				t_y.push_back(i);
			meshgrid(t_x, t_y, X, Y);
		}
			break;
		default:
			break;
		}
	}

	/**
	 * Matlab Rectangular Grid
	 * See http://www.mathworks.nl/help/matlab/ref/meshgrid.html
	 */
	template<typename T, typename U>
	static void meshgrid(const std::vector<T> &rx, const std::vector<T> &ry, U &X, U &Y)
	{
		cv::repeat(cv::Mat(rx).reshape(1, 1), ry.size(), 1, X);
		cv::repeat(cv::Mat(ry).reshape(1, 1).t(), 1, rx.size(), Y);
	}

	static void meshgrid(const std::vector<std::vector<double> > &, std::vector<cv::Mat*> &);

	static cv::Mat fftshift(const cv::Mat &, int = 0);
	static void whiten(const cv::Mat &, const cv::Size &, cv::Mat &);
	static void whiten(const cv::Mat &, const std::vector<int> &, cv::Mat &);

	static void get_images_canvas(const int &, const cv::Mat &, const cv::Size &, cv::Mat &);
	static void get_images_canvas(const cv::Size &, const cv::Mat &, const cv::Size &, cv::Mat &);
	static void get_heatmap(const cv::Mat&, cv::Mat &, int = 10, int = 0, int = cv::COLORMAP_JET, bool = true);
};

} /* namespace nl_uu_science_gmt */
#endif /* UTILITY_H_ */
