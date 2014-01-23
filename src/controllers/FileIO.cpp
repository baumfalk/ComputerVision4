/*
 * IO.cpp
 *
 *  Created on: Mar 26, 2013
 *      Author: coert
 *
 * This is completely without warranty. If you mess up your data using functions here, I'm not responsible!
 */

#include "FileIO.h"

namespace bf = boost::filesystem;

namespace nl_uu_science_gmt
{

long FileIO::getFileSize(const std::string &file_name)
{
	FILE * pFile;
	long size;
	pFile = fopen(file_name.c_str(), "rb");
	if (pFile == NULL)
		return -1;
	else
	{
		fseek(pFile, 0, SEEK_END);
		size = ftell(pFile);
		fclose(pFile);
		return size;
	}
}

std::string FileIO::getFileBasedir(const std::string &file_path)
{
	bf::path p(file_path);
	return p.parent_path().string();
}

std::string FileIO::getFileBasename(const std::string &file_name)
{
	return bf::basename(file_name);
}

std::string FileIO::getFileExtension(const std::string &file_name)
{
	return bf::extension(file_name);
}

bool FileIO::copyFile(const std::string &file_name, const std::string &file_name2)
{
	int length;
	char data;
	std::ifstream input(file_name.c_str(), std::ifstream::binary);
	std::ofstream output(file_name2.c_str(), std::ofstream::binary);
	if (!input.is_open()) return false;
	input.seekg(0, std::ios::end);
	length = input.tellg();
	input.seekg(0, std::ios::beg);
	for (int a = 0; a < length; ++a)
	{
		input.read(&data, 1);
		output.write(&data, 1);
	}
	input.close();
	output.close();
	return true;
}

/**
 * Check if file exists
 *
 * @param file_name
 * @return boolean
 */
bool FileIO::isFile(const std::string &file_name)
{
	return bf::is_regular_file(file_name);
}

/**
 * Check if directory exists
 *
 * Windows will probably need to do something different
 */
bool FileIO::isDirectory(const std::string &path)
{
	return bf::is_directory(path);
}

std::vector<std::string> FileIO::explode(const std::string &sep, std::string input)
{
	std::vector<std::string> output;
	size_t found = input.find(sep);

	while (found != std::string::npos)
	{
		if (found > 0) output.push_back(input.substr(0, found));

		input = input.substr(found + sep.length());
		found = input.find(sep);
	}
	if (input.length() > 0) output.push_back(input);

	return output;
}

/**
 * Create directory
 *
 * Windows will probably need to do something different
 */
bool FileIO::createDirectory(const std::string &path, const bool parents)
{
	std::vector<std::string> dirs = explode(PATH_SEP, path);
	std::string fullpath = path.substr(0, 1).compare(PATH_SEP) == 0 ? PATH_SEP : "";

	int val = 0;

	if (parents && dirs.size() > 1)
	{
		for (size_t i = 0; i < dirs.size(); ++i)
		{
			fullpath += dirs[i] + PATH_SEP;

			if (!isDirectory(fullpath))
			{
				boost::filesystem::path dir(fullpath);
				val = bf::create_directory(dir);

				if (val == -1)
				{
					std::cerr << "Unable to create " << path << std::endl;
					break;
				}
			}
		}
	}
	else
	{
		boost::filesystem::path dir(fullpath);
		val = bf::create_directory(dir);
	}

	return val == 0;
}

/**
 * Method: getdir
 * Goal:   read files in directory
 *
 * For windows see:
 * http://stackoverflow.com/questions/612097/how-can-i-get-a-list-of-files-in-a-directory-using-c-or-c?answertab=active#tab-top
 * http://www.softagalleria.net/download/dirent/
 *
 * FIXME: This method ALWAYS skips hidden files ("^\..*")
 */
void FileIO::getDirectory(const std::string &path, std::vector<std::string> &files, const std::string &mask,
		const std::string prefix, const std::string postfix, const bool is_dir_only)
{
#ifndef _WIN32
	DIR *dirstream;
	struct dirent *entry;

	if ((dirstream = opendir(path.c_str())) == NULL)
	{
		std::cerr << "No directory: " + path << std::endl;
		return;
	}

	while ((entry = readdir(dirstream)) != NULL)
	{
		if (is_dir_only && entry->d_type == DT_DIR && std::string(entry->d_name).compare(0, 1, ".") != 0)
		{
			if (!mask.empty() && boost::regex_match(entry->d_name, boost::regex(mask)))
				files.push_back(prefix + std::string(entry->d_name) + postfix);
		}
		else if (!is_dir_only && std::string(entry->d_name).compare(0, 1, ".") != 0)
		{
			if (!mask.empty() && boost::regex_match(entry->d_name, boost::regex(mask)))
				files.push_back(prefix + std::string(entry->d_name) + postfix);
		}
	}

	closedir(dirstream);
#else
	//open a directory the WIN32 way
	HANDLE hFind = INVALID_HANDLE_VALUE;
	WIN32_FIND_DATA fdata;
	std::string p_path = path;

	if(path[path.size()-1] == char(PATH_SEP))
	{
		p_path = std::string(path.substr(0,path.size()-1));
	}

	hFind = FindFirstFile(p_path.append("\\*").c_str(), &fdata);

	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (!(std::string(fdata.cFileName) == "." || std::string(fdata.cFileName) == ".."))
			{
				if (is_dir_only && (fdata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					if (!mask.empty() && boost::regex_match(fdata.cFileName, boost::regex(mask)))
					files.push_back(prefix + std::string(fdata.cFileName) + postfix);
				}
				else if (!is_dir_only)
				{
					if (!mask.empty() && boost::regex_match(fdata.cFileName, boost::regex(mask)))
					files.push_back(prefix + std::string(fdata.cFileName) + postfix);
				}
			}
		}
		while (FindNextFile(hFind, &fdata) != 0);
	}
	else
	{
		std::cerr << "can't open directory\n";
		return;
	}

	if (GetLastError() != ERROR_NO_MORE_FILES)
	{
		FindClose(hFind);
		std::cerr << "some other error with opening directory: " << GetLastError() << std::endl;
		return;
	}

	FindClose(hFind);
	hFind = INVALID_HANDLE_VALUE;
#endif
	sort(files.begin(), files.end(), strcmp);
}

} /* namespace nl_uu_science_gmt */
