#include <string>
#include <unordered_set>

struct MessageFields {
    std::string headers;
    std::string sender;
    std::string messageHeader;
    std::string messageBody;
    std::unordered_multiset<std::string> links;
    std::unordered_multiset<std::string> attachments;
};