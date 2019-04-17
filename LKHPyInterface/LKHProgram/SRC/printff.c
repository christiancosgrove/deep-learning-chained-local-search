#include <stdio.h>
#include <stdarg.h>
#include <LKH.h>

// void printff(const char *fmt, ...);

/*
 * The printff function prints a message and flushes stdout.
 */

void printff(const char *fmt, ...)
{
    if (PRINT_DEBUG) {
      va_list args;

      va_start(args, fmt);
      vprintf(fmt, args);
      va_end(args);
      fflush(stdout);
    }
}
