// Meow!

#pragma once

#include <iostream> // NOLINT
#include "stdio.h"


#define COLOR_ERROR "\033[31m"
#define COLOR_RESET "\033[0m"

#define MEOW_LOG(...)                                                                                                  \
	do {                                                                                                           \
		printf("%sm %s", COLOR_ERROR, COLOR_RESET);                                                            \
		printf(__VA_ARGS__);                                                                                   \
		printf("\n");                                                                                          \
	} while (false)
