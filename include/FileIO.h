/*
 * FileIO.h
 *
 *  Created on: Mar 26, 2013
 *      Author: coert
 */

#ifndef FileIO_H_
#define FileIO_H_

#include <ctime>
#include <cstring>
#ifdef __linux__
#include <dirent.h>
#elif _WIN32
#include <Windows.h>
#endif
#include <errno.h>
#include <iostream>
#include <fcntl.h>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>

#include "Utility.h"

// Windows path separators are rather ugly
#ifdef __linux__
#define PATH_SEP "/"
#elif defined __APPLE__
#define PATH_SEP "/"
#elif defined _WIN32
#define PATH_SEP "\\"
#endif

namespace nl_uu_science_gmt
{

class FileIO
{
	/**
	 * String comparator for sorting
	 */
	static inline bool strcmp(const std::string &left, const std::string &right)
	{
		return left < right;
	}

	static std::vector<std::string> explode(const std::string &, std::string);

public:
	static void getDirectory(const std::string &, std::vector<std::string> &, const std::string &, const std::string = "",
			const std::string = "", const bool = false);
	static std::string getFileBasedir(const std::string &);
	static std::string getFileBasename(const std::string &);
	static std::string getFileExtension(const std::string &);
	static bool createDirectory(const std::string &, const bool = true);
	static bool copyFile(const std::string &, const std::string &);
	static long getFileSize(const std::string &);
	static bool isFile(const std::string &);
	static bool isDirectory(const std::string &);
};

} /* namespace nl_uu_science_gmt */
#endif /* FileIO_H_ */
