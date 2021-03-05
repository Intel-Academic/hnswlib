#pragma once
#include <filesystem>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <cstdio>

std::filesystem::path executable_directory() {
    //TODO there may be better ways to do this
    char temp[32];
    char buf[PATH_MAX];
    sprintf(temp, "/proc/%d/exe", getpid());
    int bytes = std::min(readlink(temp, buf, PATH_MAX), (ssize_t)PATH_MAX - 1);
    if(bytes >= 0)
            buf[bytes] = '\0';
    return std::filesystem::path(std::string(buf)).parent_path();
}

std::filesystem::path working_directory() {
    return std::filesystem::current_path();
}

std::string git_revision(std::filesystem::path &repo) {
    //TODO might be better ways to do this too
    std::ostringstream cmd;
    std::string temp_name = std::tmpnam(nullptr);
    cmd << "git -C " << repo << " rev-parse --short HEAD > " << temp_name;
    if (std::system(cmd.str().c_str())) {
        throw std::runtime_error("Couldn't get git revision for repo " + repo.string());
    }
    std::string rev;
    {
        std::ifstream f(temp_name);
        if (!f.good()) {
            throw std::runtime_error("Couldn't open rev file");
        }
        if (!std::getline(f, rev)) {
            throw std::runtime_error("git rev empty");
        }
    }
    cmd.str(std::string());
    cmd.clear();
    cmd << "git -C " << repo << " status -s > " << temp_name;
    if (std::system(cmd.str().c_str())) {
        std::cerr << cmd.str() << std::endl;
        throw std::runtime_error("Couldn't get git status for repo " + repo.string());
    }
    std::string st;
    {
        std::ifstream f(temp_name);
        std::getline(f, st);
        std::filesystem::remove(temp_name);
    }
    if (!st.empty()) {
        rev += "+uncommitted";
    }
    return rev;
}

std::string machine_name() {
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    return std::string(hostname);
}

std::string date_time() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(gmtime(&now_time), "%FT%TZ");
    return ss.str();
}
