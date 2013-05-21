#include <string>

#define EAX 0
#define EBX 1
#define ECX 2
#define EDX 3

#ifdef _MSC_VER

#include <intrin.h>

#else // g++/clang

static void __cpuid(int reg[4], int type)
{
  __asm__ __volatile__ (
    "cpuid": "=a" (reg[EAX]), "=b" (reg[EBX]), "=c" (reg[ECX]), "=d" (reg[EDX]) : "a" (type)
  );
}

#endif

static bool test_bit(int value, int bit)
{
  return (value & (1 << bit)) != 0;
}

int main(int argc, char** argv)
{
  if (argc < 2)
    return -1;

  std::string arg = argv[1];

  int reg[4];

  __cpuid(reg, 1);

  if (arg == "mmx")
    return test_bit(reg[EDX], 23);
  if (arg == "sse")
    return test_bit(reg[EDX], 25);
  if (arg == "sse2")
    return test_bit(reg[EDX], 26);
  if (arg == "sse3")
    return test_bit(reg[ECX], 0);
  if (arg == "ssse3")
    return test_bit(reg[ECX], 9);
  if (arg == "sse4.1")
    return test_bit(reg[ECX], 19);
  if (arg == "sse4.2")
    return test_bit(reg[ECX], 20);
  if (arg == "avx")
    return test_bit(reg[ECX], 28);

  return -1;
}
