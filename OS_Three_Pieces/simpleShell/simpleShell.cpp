#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>
#include <vector>
#include <string>
using namespace std;

vector<string> PATH;

char* buf = NULL;
size_t len = 0;
char delim = ' ';

//! To parse the input
int parse(FILE *fp, char *argv[]);
//! To deal with the built-in command
bool builtInCommandParse(int argc, char* argv[]);
//! To deal with the other command
void otherCommandParse(char *argv[]);
//! To output the error information
void error();

int main(int argc, char *argv[]) {
    
    FILE *fp;
    
    bool isInteractive;
    //* To determine the mode
    if(argc == 1) {
        //! Interactive mode
        isInteractive = true;
        fp = stdin;
    }
    else {
        //! Batch mode
        fp = fopen(argv[1], "rt");
        isInteractive = false;
        if(fp == NULL) {
            error();
            exit(0);
        }
    }

    //* To bgein the loop

    char *argumentV[10];

    while(true) {
        if(isInteractive)
            printf("wish> ");
        int num = parse(fp, argumentV);
        //! To parse the built-in command
        if(builtInCommandParse(num, argumentV)) ;
        else {
            otherCommandParse(argumentV);
        }
        //! To parse other command

        buf = NULL;
        len = 0;

        //! To free the allocated space
        for(int i = 0; i < num; ++i) {
            free(argumentV[i]);
        }
    }
}

int parse(FILE *fp, char *argv[]) {

    //* To get the input from fp.
    getline(&buf, &len, fp);

    /*
     * We need to deal with interactive and batch
     * When feof(fp) != 0, It meets the EOF.
     * However, for interactive, when we hit Ctrl + D, buf = "".
     * For batch, we need to take care below situation:
       !txt
       path /usr/bin
       ls
       !txt
     * when iterating `ls`, feof(fp) == 0, however, buf != "", we need to handle buf.
     ? Actually I don't find a better way to do.
    */
    if(feof(fp) != 0 ) {
        if(strcmp("", buf) == 0) {
            exit(0);
        }
    }
    
    //* To change the '\n' to '\0' in interactive mode.
    if(buf[strlen(buf) -1] == '\n')
        buf[strlen(buf) - 1] = '\0';

    //* To make a new modifiedBuf for mutiple '\t' and ' ' char.
    char modifiedBuf[strlen(buf)];
    bool isFirstSpaceOrTab = false;
    int lengthOfModifiedBuf = 0;
    for(int i = 0; buf[i] != '\0'; ++i) {
        if ((buf[i] == ' ' || buf[i] == '\t') ) {
            if(!isFirstSpaceOrTab) {
                continue;
            }
            else {
                modifiedBuf[lengthOfModifiedBuf++] = ' ';
                isFirstSpaceOrTab = false;
            }
        }
        else {
            modifiedBuf[lengthOfModifiedBuf++] = buf[i];
            isFirstSpaceOrTab = true;
        }
    }
    modifiedBuf[lengthOfModifiedBuf] = '\0';

    char *bufp = modifiedBuf;

    int argc = 0;
    while(bufp != NULL) {
        const char *s = strsep(&bufp, &delim);
        argv[argc] = new char[strlen(s)];
        strcpy(argv[argc], s);
        ++argc;
    }

    argv[argc] = NULL;
    return argc;
}

bool builtInCommandParse(int argc, char *argv[]) {
    if(strcmp(argv[0], "exit") == 0) {
        //* If the argument number == 1
        if(argc == 1 )  {
            exit(0);
        }
        else {
            error();
        }
        return true;
    }
    else if(strcmp(argv[0], "cd") == 0) {
        //* If the argument number == 1.
        if(argc == 2) {
            if(chdir(argv[1]) != 0) {
                error();
            }
        }
        else {
            error();
        }
        return true;
    }
    else if(strcmp(argv[0], "path") == 0) {
        PATH.resize(argc - 1);
        for(int i = 0; i < argc - 1; ++i) {
            PATH.at(i) = argv[i + 1];
        }
        return true;
    }
    else return false;
}

void otherCommandParse(char *argv[]) {
    pid_t pid = fork();
    if(pid == 0) {
        string findTheExecution;
        bool isFindTheExecution = false;

        for(auto &path: PATH) {
            findTheExecution = path + "/" + argv[0];
            if(access(findTheExecution.c_str(), X_OK) == 0) {
                isFindTheExecution = true;
                break;
            }
        }

        if(!isFindTheExecution) {
            error();
            //* Must use exit(0) to terminate the child process
            exit(0);
        }
        else {
            execv(findTheExecution.c_str(), argv);
        }
    }
    else if (pid > 0) {
        wait(NULL);
    }
    else {
        error();
    }
}

void error() {
    char error_message[30] = "An error has occurred\n";
    write(STDERR_FILENO, error_message, strlen(error_message)); 
}