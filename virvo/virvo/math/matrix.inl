#pragma once

#include "vector.h"

#include <cassert>


namespace virvo
{

namespace math
{


template < >
class Matrix< vector< 4, float > >
{
public:

    typedef vector< 4, float > row_type;

    inline Matrix()
    {
    }

    inline Matrix(float* data)
    {

        rows[0] = row_type( &data[ 0] );
        rows[1] = row_type( &data[ 4] );
        rows[2] = row_type( &data[ 8] );
        rows[3] = row_type( &data[12] );
        
    }

    inline row_type row(size_t i) const
    {
        assert(i < 4);
        return rows[i];
    }

    inline void setRow(size_t i, row_type const& v)
    {
        assert(i < 4);
        rows[i] = v;
    }

private:

    row_type rows[4];

};


inline vector< 4, float > operator*(Matrix< vector< 4, float > > const& m, vector< 4, float > const& v)
{

    return vector< 4, float >
    (
        dot(m.row(0), v),
        dot(m.row(1), v),
        dot(m.row(2), v),
        dot(m.row(3), v)
    );

}


} // math

} // virvo


