#include <filesystem>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;


void displayDirectoryTree(const fs::path& pathToScan, int level = 0) {
  for (const auto& entry : fs::directory_iterator(pathToScan)) {
    const auto filenameStr = entry.path().filename().string();
    if (entry.is_directory()) {
      std::cout << std::setw(level * 3) << "" << filenameStr << '\n';
      displayDirectoryTree(entry, level + 1);
    } else if (entry.is_regular_file()) {
      std::cout << std::setw(level * 3) << "" << filenameStr <<
        ", size " << fs::file_size(entry) << " bytes\n";
    } else {
      std::cout << std::setw(level * 3) << "" << " [?]" << filenameStr << '\n';
    }
  }
}

int main(int argc, char *argv[]) {
  const fs::path pathToShow{fs::current_path()};

  std::cout << "listing files in the directory: " << fs::absolute(pathToShow).string() << '\n';

  displayDirectoryTree(pathToShow);
}
