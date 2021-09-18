#ifndef TEXTANALYSIS_HPP
#define TEXTANALYSIS_HPP

#include <string>

struct TextAnalysis {
    std::string laResults;
    std::string laMessageId;
    TextAnalysis(std::string results, std::string messageId) {
        laResults = results;
        laMessageId = messageId;
    }
    TextAnalysis() {}
};

#endif // TEXTANALYSIS_HPP