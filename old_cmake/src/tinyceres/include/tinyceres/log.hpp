// Meow!

#pragma once

#include <iostream> // NOLINT
#include "stdio.h"


#define COLOR_ERROR "\033[31m"
#define COLOR_RESET "\033[0m"

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define MEOW_LOG(...)                                                                                                  \
	do {                                                                                                           \
		printf("%sm %s", COLOR_ERROR, COLOR_RESET);                                                            \
		/*printf("%s:", __func__);*/                                                                           \
		printf("[%s:", __FILENAME__);                                                                           \
		printf("%d]", __LINE__);                                                                                \
		printf(__VA_ARGS__);                                                                                   \
		printf("\n");                                                                                   \
		/*printf("\n%seow%s\n", COLOR_ERROR, COLOR_RESET);*/                                                       \
	} while (false)

#define MEOW_LOG_FN(...)                                                                                               \
	do {                                                                                                           \
		printf("%sm %s", COLOR_ERROR, COLOR_RESET);                                                            \
		printf("%s:", __FILENAME__);                                                                           \
		printf("%d\n", __LINE__);                                                                              \
		printf("%s@ %s", COLOR_ERROR, COLOR_RESET);                                                            \
		printf("%s ", __PRETTY_FUNCTION__);                                                                    \
		printf("%s@ %s", COLOR_ERROR, COLOR_RESET);                                                            \
		printf("\n");                                                                                          \
		printf(__VA_ARGS__);                                                                                   \
		printf("\n");                                                                                   \
		/*printf("\n%seow%s\n", COLOR_ERROR, COLOR_RESET);*/                                                       \
	} while (false)\
