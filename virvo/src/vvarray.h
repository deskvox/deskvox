// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_ARRAY_H
#define VV_ARRAY_H

#include <iostream>
#include <string.h>
#include "vvexport.h"

/** Templated dynamic array class.<P>
  Example usage:
  <PRE>
  vvArray<int*> test;
  int i[] = {1,2,3};
  test.append(&i[0]);
  test.append(&i[1]);
  test.insert(1, &i[2]);
  test.remove(2);
  test.clear();
  </PRE>
@author Jurgen P. Schulze
*/

using namespace std;

template<class T> class vvArray
{
  public:
    vvArray();
    vvArray(int, int);
    vvArray(const vvArray<T>&);
    ~vvArray();

    void clear();
    void set(int, const T&);
    T*   get(int);
    void append(const T&);
    void insert(int, const T&);
    void replace(int, const T&);
    void remove(int);
    void removeLast();
    bool removeElement(const T&);
    void resize(int);
    void setIncrement(int);
    void fill(const T&);
    T*   first();
    T*   last();
    int  find(const T&);
    int  count();
    void print(char*);
    T*   getArrayPtr();
    void deleteElementsNormal();
    void deleteElementsArray();

    /// Direct array access:
    T & operator[] (int index)
    {
      if (index<0 || index>(usedSize-1))
      {
        return nullValue;
      }
      return data[index];
    }

    /// Direct array access:
    const T & operator[] (int index) const
    {
      if (index<0 || index>(usedSize-1))
      {
        return nullValue;
      }
      return data[index];
    }

    /// Assign from another vvArray:
    vvArray<T> &operator =(const vvArray<T>& v);

  private:
    T*   data;                                    ///< actual data array
    int  usedSize;                                ///< number of array elements actually used
    int  allocSize;                               ///< number of array elements allocated in memory
    int  allocInc;                                ///< number of array elements by which the array grows when increased
    T    nullValue;                               ///< NULL value to use as a return value

    void incSize();
};

//----------------------------------------------------------------------------
/// Default constructor. Array is empty, allocation increment is 10 elements.
template<class T> vvArray<T>::vvArray()
{
  nullValue = (T)0;
  allocSize = 0;
  allocInc  = 10;
  usedSize   = 0;
  data  = 0;
}

//----------------------------------------------------------------------------
/** Constructor for a new array with 'amount' initial elements and an
  array increment of 'inc'.
*/
template<class T> vvArray<T>::vvArray(int amount, int inc)
{
  int i;

  nullValue = (T)0;
  allocSize = amount;
  allocInc = inc;
  usedSize = 0;
  data = new T[allocSize];
  for (i=0; i<allocSize; ++i) data[i] = 0;
}

//----------------------------------------------------------------------------
/// Copy constructor.
template<class T> vvArray<T>::vvArray(const vvArray<T>& v)
{
  int i;

  nullValue = (T)0;
  allocSize = v.allocSize;
  allocInc = v.allocInc;
  usedSize = v.usedSize;
  data = new T[allocSize];
  for (i=0; i<usedSize; ++i) data[i] = v.data[i];
}

//----------------------------------------------------------------------------
/// Destructor: free all memory
template<class T> vvArray<T>::~vvArray()
{
  clear();
}

//----------------------------------------------------------------------------
/// Return pointer to data array.
template<class T> T* vvArray<T>::getArrayPtr()
{
  return data;
}

//----------------------------------------------------------------------------
/// Append element passed directly.
template<class T> void vvArray<T>::append(const T& in_data)
{
  if (usedSize == allocSize) incSize();
  data[usedSize] = in_data;
  ++usedSize;
}

//----------------------------------------------------------------------------
/// Replace element at the given index.
template<class T> inline void vvArray<T>::replace(int index, const T& newData)
{
  if (index<0 || index>(usedSize - 1)) return;
  data[index] = newData;
}

//----------------------------------------------------------------------------
/** Insert element at the given array index. If index is out of bounds,
  nothing will be done.
*/
template<class T> inline void vvArray<T>::insert(int index, const T& in_data)
{
  int i;

  if (index < 0) return;
  if (usedSize == allocSize)
    incSize();
  if (index >= usedSize) return;

  for (i=usedSize; i>index; --i)
    data[i] = data[i - 1];

  data[index] = in_data;
  usedSize++;
}

//----------------------------------------------------------------------------
/** Remove element from array.
  If the array is a list of pointers, the elements pointed to must be deleted separately!
  @param index index of element to remove (0 for first element, etc.)
*/
template<class T> inline void vvArray<T>::remove(int index)
{
  int i;

  if (index<0 || index>(usedSize - 1)) return;

  for (i=index; i<usedSize-1; ++i)
    data[i] = data[i + 1];

  --usedSize;
}

//----------------------------------------------------------------------------
/** Remove last element. If array is empty, nothing happens.
  The allocated array size is not changed.
*/
template<class T> inline void vvArray<T>::removeLast()
{
  if (usedSize>0) --usedSize;
}

//----------------------------------------------------------------------------
/** Finds first occurrence of element 'in_data' and delete it.
  @return true if successful, otherwise false
*/
template<class T> inline bool vvArray<T>::removeElement(const T& in_data)
{
  int index = find(in_data);
  if (index != -1)
  {
    remove(index);
    return true;
  }
  else return false;
}

//----------------------------------------------------------------------------
/** Returns the element at 'index'.
  If index is out of bounds, NULL is returned.
  The current index is set to the index of the returned element.
*/
template<class T> inline T* vvArray<T>::get(int index)
{
  if (index<0 || index>=usedSize) return &nullValue;

  return &(data[index]);
}

//----------------------------------------------------------------------------
/** Returns the last array element or NULL if array is empty.
  The current index is set to the last element.
*/
template<class T> inline T* vvArray<T>::last()
{
  if (usedSize>0)
  {
    return &(data[usedSize - 1]);
  }
  else return &nullValue;
}

//----------------------------------------------------------------------------
/// Get first element and set current index to 0
template<class T> inline T* vvArray<T>::first()
{
  if (usedSize==0) return NULL;
  return &(data[0]);
}

//----------------------------------------------------------------------------
/** Find the first occurrence of the specified element.
  @element element to find in array
  @return index of the desired element, or -1 if element was not found
*/
template<class T> inline int vvArray<T>::find(const T& element)
{
  int i;

  if (usedSize==0) return -1;

  for (i=0; i<usedSize; ++i)
  {
    if (data[i] == element) return i;
  }

  return -1;
}

//----------------------------------------------------------------------------
/** Resize the array.
  @param newSize new array size [elements]
*/
template<class T> void vvArray<T>::resize(int newSize)
{
  int i;
  T* newData;

  newData = new T[allocSize];
  allocSize = newSize;

  // Copy old array:
  memcpy(newData, data, usedSize * sizeof(T));

  delete[] data;
  data = newData;
}

//----------------------------------------------------------------------------
/** Set the array incrementation size [elements].
 */
template<class T> void vvArray<T>::setIncrement(int inc)
{
  allocInc = inc;
}

//----------------------------------------------------------------------------
/// Fills the whole array with fillData.
template<class T> void vvArray<T>::fill(const T& fillData)
{
  int i;

  for (i=0; i<usedSize; ++i)
    data[i] = fillData;
}

//----------------------------------------------------------------------------
/// Clear the array (frees all data).
template<class T> void vvArray<T>::clear()
{
  delete[] data;
  data = 0;
  usedSize = 0;
  allocSize = 0;
}

//----------------------------------------------------------------------------
/** '=' copies the array.
  Only the array entries are copied, but not the elements the pointers
  point to in case of a pointer array!
*/
template<class T> vvArray<T> &vvArray<T>::operator =(const vvArray<T>& v)
{
  if (this != &v)
  {
    clear();
    usedSize = v.usedSize;
    allocSize = v.allocSize;
    data = new T[allocSize];
    memcpy(data, v.data, usedSize * sizeof(T));
  }

  return *this;
}

//----------------------------------------------------------------------------
/// Set array element. Any previously existing element will be overwritten.
template<class T> inline void vvArray<T>::set(int index, const T& newData)
{
  if (usedSize == 0 || index < 0 || index > (usedSize - 1)) return;
  data[index] = newData;
}

//----------------------------------------------------------------------------
/// Returns the number of array enries.
template<class T> inline int vvArray<T>::count()
{
  return usedSize;
}

//----------------------------------------------------------------------------
/** Print out the contents of the array.
  @param title some string to print before the array
*/
template<class T> inline void vvArray<T>::print(char* title)
{
  int i;

  cerr << title << endl;

  if (usedSize == 0)
  {
    cerr << "empty array" << endl;
    return;
  }

  for (i=0; i<usedSize; ++i)
    cerr << "[" << i << "]: " << data[i] << endl;
}

//----------------------------------------------------------------------------
/** Increase allocated size: create a new array and copy all elements from
  the old array.
*/
template<class T> void vvArray<T>::incSize()
{
  allocSize += allocInc;
  T* newData = new T[allocSize];

  // Copy old array:
  memcpy(newData, data, usedSize * sizeof(T));

  delete[] data;
  data = newData;
}

//----------------------------------------------------------------------------
template<class T> void vvArray<T>::deleteElementsNormal()
{
  int i;
  for (i=0; i<usedSize; ++i)
  {
    delete data[i];
  }
}

//----------------------------------------------------------------------------
template<class T> void vvArray<T>::deleteElementsArray()
{
  int i;
  for (i=0; i<usedSize; ++i)
  {
    delete[] data[i];
  }
}
#endif
