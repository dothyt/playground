#include <iostream>
#include <string>
#include <utility>

void overloaded_function(std::string& param) {
  std::cout << "std::string& version" << std::endl;
}
void overloaded_function(std::string&& param) {
  std::cout << "std::string&& version" << std::endl;
}

template<typename T>
void pass_through(T&& param) {
  overloaded_function(std::forward<T>(param));
}

int main() {
  std::string pes;
  pass_through(pes);
  pass_through(std::move(pes));
  pass_through("string literal");
}