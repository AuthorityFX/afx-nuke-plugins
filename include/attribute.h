// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Ryan P. Wilson
//
//      Authority FX, Inc.
//      www.authorityfx.com

#ifndef AFX_ATTRIBUTE_H_
#define AFX_ATTRIBUTE_H_

#include <boost/ptr_container/ptr_list.hpp>

#include <assert.h>

#include <type_traits>

#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>

namespace afx {

// Attribute to store adhoc info like thread ids, channel names, etc
struct Attribute {
  std::string name;
  int value;
  Attribute(std::string name, int value) : name(name), value(value) {}
};

class AttributeBase {
 public:
  void AddAttribute(const std::string& name, int value) { attributes_.push_back(Attribute(name, value)); }
  void AddAttributes(std::vector<Attribute> attributes) {
    attributes_.insert(attributes_.end(), attributes.begin(), attributes.end());
  }
  int GetAttribute(const std::string& name) const {
    auto it = std::find_if(attributes_.begin(), attributes_.end(), [name](const Attribute& attr) { return attr.name.compare(name) == 0; });
    if (it != attributes_.end()) {
      return it->value;
    } else {
      throw std::out_of_range(std::string("No attribute named ") + name);
    }
  }

protected:
  std::vector<Attribute> attributes_;
};

template <typename T>
class Array {
 public:
  typedef typename boost::ptr_list<T>::iterator ptr_list_it;

  virtual void Add() { array_.push_back(new T()); }
  void Clear() { array_.clear(); }
  T* GetBackPtr() { return &array_.back(); }
  T* GetPtrByAttribute(const std::string& name, int value) {
    ptr_list_it it;
    for (it = array_.begin(); it != array_.end(); ++it) {
      if (it->GetAttribute(name) == value) { break; }
    }
    if (it != array_.end()) {
      return &(*it);
    } else {
      std::stringstream ss;
      ss << "afx::Array - no image with attribute '" << name << "' = " << value;
      throw std::out_of_range(ss.str());
    }
  }
  T* GetPtrByAttributes(std::vector<Attribute> list) {
    ptr_list_it it;
    for (it = array_.begin(); it != array_.end(); ++it) {
      unsigned int num_found = 0;
      for (auto a_it = list.begin(); a_it != list.end(); ++a_it) {
        if (it->GetAttribute(a_it->name) == a_it->value ) { num_found++; }
      }
      if (num_found == list.size()) { break; }
    }
    if (it != array_.end()) {
      return &(*it);
    } else {
      std::stringstream ss;
      ss << "afx::Array - No image with attributes: ";
      for (auto a_it = list.begin(); a_it != list.end(); ++a_it) {
        ss << "'" << a_it->name << "'' = " << a_it->value << "'";
        if (a_it + 1 != list.end()) { ss << ", "; }
      }
      throw std::out_of_range(ss.str());
    }
  }
  // Check if attribute exists
  bool HasAttribute(const std::string& name, int value) {
    bool found = false;
    for (ptr_list_it it = array_.begin(); it != array_.end(); ++it) {
      if (it->GetAttribute(name) == value) {
        found = true;
        break;
      }
    }
    return found;
  }
  bool HasAttributes(std::vector<Attribute> list) {
    bool found = false;
    for (ptr_list_it it = array_.begin(); it != array_.end(); ++it) {
      unsigned int num_found = 0;
      for (auto a_it = list.begin(); a_it != list.end(); ++a_it) {
        if (it->GetAttribute(a_it->name) == a_it->value ) { num_found++; }
      }
      if (num_found == list.size()) {
        found = true;
        break;
      }
    }
    return found;
  }
  ptr_list_it GetBegin() { return array_.begin(); }
  ptr_list_it GetEnd() { return array_.end(); }
  ptr_list_it GetRBegin() { return array_.rbegin(); }
  ptr_list_it GetREnd() { return array_.rend(); }

private:
  static_assert(std::is_base_of<AttributeBase, T>::value, "T must inherit from AttributeBase");
protected:
  boost::ptr_list<T> array_;
};

}  // namespace afx

#endif  // AFX_ATTRIBUTE_H_
