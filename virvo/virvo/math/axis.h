#ifndef VV_MATH_AXIS_H
#define VV_MATH_AXIS_H


namespace virvo
{

template < int Dim >
class cartesian_axis;


template < >
class cartesian_axis< 2 >
{
public:

    enum label { X, Y };

    /* implicit */ cartesian_axis(label l) : val_(l) {}

    operator label() const { return val_; }

private:

    label val_;

};


template < >
class cartesian_axis< 3 >
{
public:

    enum label { X, Y, Z };

    /* implicit */ cartesian_axis(label l) : val_(l) {}

    operator label() const { return val_; }

private:

    label val_;

};


} // virvo


#endif // VV_MATH_AXIS_H


