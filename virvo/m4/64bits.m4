dnl Checks if the size of a void pointer is 64 bits
dnl
AC_DEFUN([CHECK_64BITS], [

AC_TRY_RUN([
int main(void)
{
    return sizeof(void *) == 8;
}
], [ac_64bit_pointer=no], [ac_64bit_pointer=yes], 
    [ac_64bit_pointer=yes])

if test "$ac_64bit_pointer" = "yes"; then
    ARCH_64BITS=yes
fi
])
