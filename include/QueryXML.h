/*
 * QueryXML.h
 *
 *  Created on: Feb 12, 2013
 *      Author: coert
 */

#ifndef QUERYXML_H_
#define QUERYXML_H_

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include "../modules/ticpp/ticpp.h"

namespace nl_uu_science_gmt
{

typedef ticpp::Element* QXmlElmPtr;
typedef std::vector<QXmlElmPtr> QXmlElms;
typedef std::vector<QXmlElmPtr>::const_iterator QXmlCIt;

class QueryXML
{
	const std::string _file;
	ticpp::Document _doc;

	static const enum Exceptions
	{
		BAD_QUERY, KEY_NOT_FOUND, ATTRIBUTE_NOT_FOUND, FILE_NOT_FOUND, UNREADABLE_VALUE
	} _exc;

	void showError(const Exceptions&, const std::string&) const;

	QXmlElmPtr getElement(ticpp::Node*, std::vector<std::string> keys) const;
	QXmlElmPtr process(ticpp::Node*, const std::string&) const;
	void processKey(const std::string&, std::vector<std::string>&, std::string&) const;

	static inline std::vector<std::string>& split(const std::string& s, char delim, std::vector<std::string>& elems)
	{
		std::stringstream ss(s);
		std::string item;
		while (std::getline(ss, item, delim))
			elems.push_back(item);
		return elems;
	}

	static inline std::vector<std::string> split(const std::string& s, char delim)
	{
		std::vector<std::string> elems;
		return split(s, delim, elems);
	}

	template<typename U, typename T>
	inline const U toValue(T* node) const
	{
		U value;
		node->GetValue(&value);
		return value;
	}

	inline const bool toEmpty(bool* v = NULL) const
	{
		return false;
	}

	inline const std::string toEmpty(std::string* v = NULL) const
	{
		return "";
	}

	template<typename T>
	inline const T toEmpty(T* v = NULL) const
	{
		return 0;
	}

public:
	QueryXML(const std::string &);
	virtual ~QueryXML();

	bool hasValue(const std::string&, QXmlElmPtr = NULL) const;

	/**
	 * Get the value of type <T> from the given key relative to node
	 *
	 * @param key
	 * @param node
	 * @return
	 */
	template<typename T>
	inline const T getValue(const std::string& key, QXmlElmPtr el = NULL) const
	{
		if (el == NULL) el = (QXmlElmPtr) &_doc;

		std::vector<std::string> keys;
		std::string attribute;

		try
		{
			processKey(key, keys, attribute);
			QXmlElmPtr element = getElement(el, keys);

			if (!element)
			{
				std::cout << "Warning: xml-key '" << key << "' not found!" << std::endl;
				return toEmpty((T*) NULL);
			}

			if (attribute.length() > 0)
			{
				ticpp::Iterator<ticpp::Attribute> attrib;
				for (attrib = attrib.begin(element); attrib != attrib.end(); attrib++)
					if (attrib->Name().compare(attribute) == 0) return toValue<T>(attrib.Get());
			}
			else
			{
				ticpp::Iterator<ticpp::Text> child;
				child = child.begin(element);

				if (child == NULL)
				{
					return toEmpty((T*) NULL);
				}
				else
				{
					return toValue<T>(child.Get());
				}
			}

		} catch (Exceptions& exception)
		{
			showError(exception, key);
		}

		std::cerr << "Bad entry for " << _file << "::" << key << std::endl;
		return toEmpty((T*) NULL);
	}

	/**
	 * Get the values from path-string, starting at Node*
	 */
	const QXmlElms getValues(const std::string&, QXmlElmPtr = NULL) const;
};

} /* namespace nl_uu_cs_gmt */
#endif /* QUERYXML_H_ */
