#ifndef DATA_HPP
#define DATA_HPP

#include <string>
#include <vector>
#include "../Data/AttachmentAnalysis.hpp"
#include "../Data/TextAnalysis.hpp"
#include "../Data/HeaderAnalysis.hpp"
#include "../Data/LinksAnalysis.hpp"
using namespace std;

struct Results {    
    int numberOfActivityWaiting;
    HeaderAnalysis headerAnalysisResults;
    vector<LinksAnalysis> linksAnalysisResults;
    TextAnalysis textAnalysisResults;
    vector<AttachmentAnalysis> attachmentAnalysisResults;
    Results(
        int num, HeaderAnalysis haResults,
        vector<LinksAnalysis> laResults, 
        TextAnalysis taResults,
        vector<AttachmentAnalysis> aaResults)
        : numberOfActivityWaiting(num),
          headerAnalysisResults(haResults), linksAnalysisResults(laResults),
          textAnalysisResults(taResults), attachmentAnalysisResults(aaResults){}
    Results() {}
};

#endif // DATA_HPP