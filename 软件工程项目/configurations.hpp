

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include "util.hpp"
#include <map>
#include <fstream>

using namespace std;
using namespace cv;

// set config const means the var will only be set once at the beginning, value updating won't affect the variable
#define SET_CONFIG_DOUBLE_CONST(var,defVal) var=ConfigurationVariables::GetDouble(#var,defVal);
#define SET_CONFIG_INT_CONST(var,defVal) var=ConfigurationVariables::GetInt(#var,defVal);
#define SET_CONFIG_BOOL_CONST(var,defVal) var=ConfigurationVariables::GetBool(#var,defVal);
// set config variable means the var will be updated every time rereading the configuration file
#define SET_CONFIG_DOUBLE_VARIABLE(var,defVal) var = defVal; ConfigurationVariables::GetUpdatingDouble(#var,&var);
#define SET_CONFIG_INT_VARIABLE(var,defVal) var = defVal; ConfigurationVariables::GetUpdatingInt(#var,&var);
#define SET_CONFIG_BOOL_VARIABLE(var,defVal) var = defVal; ConfigurationVariables::GetUpdatingBool(#var,&var);

#define DEBUG_MODE ConfigurationVariables::DebugMode

class ConfigurationVariables
{
public:
    static bool DebugMode, Loaded,KeepUpdateConfiguration;
    static map<string, string> itemMap;
    static int MainEntry ;
    static int resolutionType ;          // 0:640x480   1:1280x720   2:1920x1080

    static int resWidth,resHeight;

    static void ReadConfiguration(bool init = true, string path = FILEDIR(config.ini))
    {
        try {
            DebugMode = true;

            ifstream inputfile(path);
            string line, key, value;
            while (getline(inputfile, line))
            {
                key = "";
                value = "";
                bool state = false;
                FOREACH(i, line.length())
                {
                    if (line[i] == '=') state = true;
                    else if (line[i] == '#') break;
                    else if (state) value += line[i];
                    else key += line[i];
                }
                if (state)
                {
                    // add only when variable unexist, otherwise update the value
		    if (!TryModifyExistedVariables(key,value) && GetString(key) == "UnknownKey")
                        itemMap.insert(pair<string, string>(key, value));
                }
            }
            inputfile.close();
            Loaded = true;
            if (init)
            {
                SET_CONFIG_BOOL_VARIABLE(DebugMode,true);
                SET_CONFIG_INT_VARIABLE(MainEntry,0);
                SET_CONFIG_INT_CONST(resolutionType,0);
                SET_CONFIG_BOOL_CONST(KeepUpdateConfiguration,true);
            }
            // set resolution
            if (resolutionType == 1)
            {
                resWidth = 1280; resHeight = 720;
            }
            else if (resolutionType == 2)
            {
                resWidth = 1920; resHeight = 1080;
            }
            else // default 640x480
            {
                resWidth = 640; resHeight=480;
            }
        }
        catch (...) {
            Loaded = false;
        }
    }

    static string GetString(string key)
    {
        map<string, string>::iterator iter = itemMap.find(key);
        if (iter != itemMap.end()) return (*iter).second;
        else return "UnknownKey";
    }

    static int GetInt(string key, int defInt = 0)
    {
        string str = GetString(key);
        sscanf(str.c_str(), "%d", &defInt);
        return defInt;
    }

    static double GetDouble(string key, double defDouble = 0)
    {
        string str = GetString(key);
        sscanf(str.c_str(), "%lf", &defDouble);
        return defDouble;
    }

    static bool GetBool(string key, bool defBool)
    {
        return GetInt(key,defBool ? 1 : 0) != 0;
    }

    static void GetUpdatingInt(string key,int *var)
    {
	string str = GetString(key);
        sscanf(str.c_str(), "%d", var);
        int_variables.push_back(pair<string,int*>(key,var));
    }

    static void GetUpdatingDouble(string key,double *var)
    {
	string str = GetString(key);
        sscanf(str.c_str(), "%lf", var);
        double_variables.push_back(pair<string,double*>(key,var));
    }

    static void GetUpdatingBool(string key,bool *var)
    {
	string str = GetString(key);
	int t = (*var);
        sscanf(str.c_str(), "%d", &t);
        (*var) = t;
        bool_variables.push_back(pair<string,bool*>(key,var));
    }

    /*
     *   Check if the variable exists, if so update the value and return true,
     * or return false and do nothing.
     */
    static bool TryModifyExistedVariables(string key,string val)
    {
        FOREACH(i,int_variables.size()) if (int_variables[i].first == key)
        {
            sscanf(val.c_str(),"%d",int_variables[i].second);
            return true;
        }
        FOREACH(i,double_variables.size()) if (double_variables[i].first == key)
        {
            sscanf(val.c_str(),"%lf",double_variables[i].second);
            return true;
        }
        FOREACH(i,bool_variables.size()) if (bool_variables[i].first == key)
        {
            int t = *(bool_variables[i].second) ? 1 : 0;
            sscanf(val.c_str(),"%d",&t);
	    *(bool_variables[i].second) = t;
            return true;
        }
        return false;
    }
protected:
    static vector<pair<string,int*>> int_variables;
    static vector<pair<string,double*>> double_variables;
    static vector<pair<string,bool*>> bool_variables;
    
};

map<string, string> ConfigurationVariables::itemMap;
bool ConfigurationVariables::DebugMode, ConfigurationVariables::Loaded,ConfigurationVariables::KeepUpdateConfiguration;
int ConfigurationVariables::MainEntry = 0;
int ConfigurationVariables::resolutionType = 0;
int ConfigurationVariables::resWidth = 640;
int ConfigurationVariables::resHeight = 480;

vector<pair<string,int*>> ConfigurationVariables::int_variables;
vector<pair<string,double*>> ConfigurationVariables::double_variables;
vector<pair<string,bool*>> ConfigurationVariables::bool_variables;
