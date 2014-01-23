/*
 * QueryXML.cpp
 *
 * document.xml:
 * <?xml version="1.0" encoding="UTF-8"?>
 * <root>
 * 	<values type="integer">
 * 		<x>1</x>
 * 		<x>2</x>
 * 		<x name="three">3</x>
 * 		<x>4</x>
 * 	</values>
 * </root>
 *
 * QueryXML xml("/home/coert/document.xml");
 * bool isValue = xml.hasValue("root");
 * int x = xml.getValue<int>("root.values.x"); // x == 1
 * string type = xml.getValue<string>("root.values@type"); // type == "integer"
 * QXmlElms xs = xml.getValues("root.values"); // xs == { QXmlElmPtr, QXmlElmPtr, QXmlElmPtr, QXmlElmPtr }
 * int x2 = xml.getValue<int>("", xs[1]); // x == 2
 * string name = xml.getValue<string>("@name", xs[2]); // name == "three"
 *
 *  Created on: Jun 25, 2011
 *      Author: coert
 */

#include "QueryXML.h"

using namespace std;
using namespace ticpp;
using namespace nl_uu_science_gmt;

namespace bf = boost::filesystem;

namespace nl_uu_science_gmt
{

QueryXML::QueryXML(const string &f) :
		_file(f)
{
	_doc.LoadFile(f);
}

QueryXML::~QueryXML()
{
}

bool QueryXML::hasValue(const std::string& key, QXmlElmPtr el) const
{
	if (el == NULL) el = (QXmlElmPtr) &_doc;

	std::vector<std::string> keys;
	std::string attribute;

	try
	{
		processKey(key, keys, attribute);
		QXmlElmPtr element = getElement(el, keys);

		if (!element)
			return false;
		else
			return true;

	} catch (Exceptions& exception)
	{
		showError(exception, key);
	}

	return false;
}

const QXmlElms QueryXML::getValues(const string& key, QXmlElmPtr element) const
{
	QXmlElms elements;

	/*
	 * Parse from root if no node is given
	 */
	if (element == NULL) element = (QXmlElmPtr) &_doc;

	try
	{
		vector<string> keys;
		string attribute;
		processKey(key, keys, attribute);

		for (Node* child = getElement(element, keys); child; child = child->NextSibling(false))
		{
			if (child->Type() == TiXmlNode::ELEMENT && child->Value().compare(keys.back()) == 0)
			{
				elements.push_back((QXmlElmPtr) child->ToElement());
			}
		}

	} catch (Exceptions& exception)
	{
		showError(exception, key);
	}

	return elements;
}

/**
 * Descend into the node to the last key in keys
 *
 * @param node
 * @param keys
 * @return
 */
QXmlElmPtr QueryXML::getElement(Node* node, vector<string> keys) const
{
	string value_key = "";

	Node* parent = node;
	for (size_t i = 0; i < keys.size(); ++i)
	{
		value_key = keys[i];
		Node* child = process(parent, value_key);

		if (child != NULL)
		{
			parent = child;
		}
		else
		{
			return NULL;
		}
	}

	if (parent->Type() == TiXmlNode::ELEMENT)
		return parent->ToElement();
	else
		return NULL;
}

/**
 * Process sibling children of the given node, return first child named 'key'
 *
 * @param pParent
 * @param key
 * @return
 */
QXmlElmPtr QueryXML::process(Node* pParent, const string& key) const
{
	if (!pParent) return NULL;

	Iterator<Node> child(key);
	//FIXME! This doens't go well with CMAKE seperate library! Don't know why!
	child = child.begin(pParent);

	if (child != child.end())
	{
		try
		{
			return child->ToElement();
		} catch (ticpp::Exception& exception)
		{
			cout << __FUNCTION__ << "@" << __LINE__ << ": --warning-- xml-key '" << key << "' generated an error!" << endl;
			return NULL;
		}
	}
//	else {
//		cout << __FUNCTION__ << "@" << __LINE__ << ": --warning-- xml-key '" << key << "' not found!" << endl;
//	}

	return NULL;
}

/**
 * Create a vector of keys that descend into XML and if exists the attribute from a key string
 * Syntax: 'root.key1.key2@attribute'
 *
 * @param key
 * @param keys
 * @param attribute
 */
void QueryXML::processKey(const string& key, vector<string>& keys, string& attribute) const
{
	/*
	 * No keys, use current node
	 */
	if (key.length() == 0) return;

	vector<string> query = split(key, '@');

	if (query.size() == 2)
	{
		attribute = query.back();
		keys = split(query.front(), '/');
	}
	else if (query.size() == 1)
	{
		keys = split(key, '/');
	}
	else
	{
		throw BAD_QUERY;
	}
}

void QueryXML::showError(const Exceptions& exc, const string& key) const
{
	stringstream error;
	error << __FILE__ << "$" << __FUNCTION__ << "@" << __LINE__ << ":(" << key << ") ";

	switch (exc)
	{
		case BAD_QUERY:
			error << "Bad query ...";
			break;
		case KEY_NOT_FOUND:
			error << "Key not found ...";
			break;
		case ATTRIBUTE_NOT_FOUND:
			error << "Attribute not found ...";
			break;
		case UNREADABLE_VALUE:
			error << "Unreadable value ...";
			break;
		default:
			error << "Unknown error ...";
			break;
	}

	cerr << error.str() << endl;

	throw exc;
}

} /* namespace nl_uu_cs_gmt */
