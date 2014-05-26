#pragma once

#include "vec4.h"


namespace virvo
{

namespace math
{


template < >
class Matrix< base_vec4< float > >
{
public:

    typedef base_vec4< float > row_type;

    inline Matrix< row_type >()
    {
    }

    inline Matrix< row_type >(virvo::Matrix const& m)
    {
        m.getRow(0, &rows[0][0], &rows[0][1], &rows[0][2], &rows[0][3]);
        m.getRow(1, &rows[1][0], &rows[1][1], &rows[1][2], &rows[1][3]);
        m.getRow(2, &rows[2][0], &rows[2][1], &rows[2][2], &rows[2][3]);
        m.getRow(3, &rows[3][0], &rows[3][1], &rows[3][2], &rows[3][3]);
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


inline base_vec4< float > operator*(Matrix< base_vec4< float > > const& m, base_vec4< float > const& v)
{

    return base_vec4< float >
    (
        dot(m.row(0), v),
        dot(m.row(1), v),
        dot(m.row(2), v),
        dot(m.row(3), v)
    );

}


} // math

} // virvo


