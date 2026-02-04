#pragma once
#include <iostream>
#include <stdexcept>
#include <string>

#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"
#define COLOR_RESET "\033[0m"

#define TRY_CATCH_COLORED(message, color)                                      \
  try {                                                                        \
    throw std::logic_error(message);                                           \
  } catch (const std::exception &e) {                                          \
    std::cerr << color << e.what() << COLOR_RESET << std::endl;                \
  } catch (...) {                                                              \
    std::cerr << color << message << ": Unknown exception occurred"            \
              << COLOR_RESET << std::endl;                                     \
  }

#define TRY_CATCH_COLORED_TERMINATE(message, color)                            \
  try {                                                                        \
    throw std::logic_error(message);                                           \
  } catch (const std::exception &e) {                                          \
    std::cerr << color << e.what() << COLOR_RESET << std::endl;                \
    std::exit(EXIT_FAILURE);                                                   \
  }

#define DebugErr(message) TRY_CATCH_COLORED_TERMINATE(message, COLOR_RED)
#define Warning(message)                                                  \
  std::cout << COLOR_YELLOW << message << COLOR_RESET << std::endl;
#define Log(message)                                                           \
  std::cout << COLOR_GREEN << message << COLOR_RESET << std::endl;

#define TRY_CATCH_CXX_COLORED(throwed, color)                                  \
  try {                                                                        \
    throwed;                                                                   \
  } catch (const std::exception &e) {                                          \
    std::cerr << color << e.what() << COLOR_RESET << std::endl;                \
    std::exit(EXIT_FAILURE);                                                   \
  }
#define CXXDebugErr(throwed) TRY_CATCH_CXX_COLORED(throwed, COLOR_RED)