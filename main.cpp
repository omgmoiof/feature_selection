//Phillip Williams
//Data sets small:58 big: 57

#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <time.h>
#include <stdlib.h>
#include <limits.h>


using namespace std;


const double EPSILON = FLT_EPSILON;
const int MAX_ENTRIES = 2048;
const int MAX_FEATURES = 64;

float data[MAX_ENTRIES][MAX_FEATURES];                   //data[r][c] r = data #, c = feature #
float normalizedData[MAX_ENTRIES][MAX_FEATURES];
int featureCount, dataSize;


void readData(string filename) //reads the data from filename to array data[][]
{
    
    ifstream ifs;
    ifs.open(filename.c_str());
    if (!ifs.is_open())
    {
        cout << "Error opening file" << endl;
        return;
        
    } 
    
    float temp;
    int r = -1;
    int c = 0;
    while (!ifs.eof())
    {
            
            ifs >> temp;
            if (fabs(temp - 1.0) < EPSILON || fabs(temp - 2.0) < EPSILON ) //if the value is 1 or 2 exactly, start a new row
            {
                r++;
                featureCount = c;
                c = 0;
                data[r][c] = temp;
            } 
            
            else
            {
                c++;
                data[r][c] = temp;

            }
    }
    
    dataSize = r + 1;
}

void normalizeData()
{
    
    float min_val[featureCount]; //minimum value of a particular feature
    float max_val[featureCount]; //maximum value of a particular feature
    
    fill(max_val, max_val + featureCount, FLT_MIN);
    fill(min_val, min_val + featureCount, FLT_MAX);
    
    for (unsigned c = 1; c < featureCount + 1; c++) //finds the max and min values for each feature
    {
        for (unsigned r = 0; r < dataSize; r++)
        {
            
            if (data[r][c] > max_val[c]) max_val[c] = data[r][c]; 
            if (data[r][c] < min_val[c]) min_val[c] = data[r][c];
        }
    }
    
    for (unsigned c = 0; c < featureCount + 1; c++) //normalizes the data from -1 to 1
    {
        for (unsigned r = 0; r < dataSize + 1; r++)
        {
            
            if (c == 0) normalizedData[r][c] = data[r][c];
            else normalizedData[r][c] = -1 + 2*(data[r][c] - min_val[c])/(max_val[c] - min_val[c]);
        }
    }
    
}

void displayData() //outputs the data set, used for testing
{
    for (unsigned r = 0; r < dataSize; r++ )
    {
        for (unsigned c = 0; c < featureCount + 1;  c++)
        {
            
            cout << normalizedData[r][c] << endl;
            
        }
        
        cout << "------------------------------" << endl;
        
        
    }
    
}


bool isElementOf(int* a, int k, int size) //checks if a particular int is within an array
{
    if (size == 0) return false;
    
    for (unsigned i = 1; i < size+1; i++)
    {
        if (a[i] == k) return true;
    }
    
    return false;
}

double leave_one_out_cross_validation(int* current_set_of_features, int k, int size, int a, int b, bool f)
{
    int correct_counter = 0;

    double accuracy;
    
    for (unsigned i = 0; i < dataSize; i++)//for each point, finds it's nearest neighbor
    {
        double minimum_distance = INT_MAX;
        
        if ( i <= a || i > b)
        {
            int nearest_neighbor = 0;
        
            for (unsigned j = 0; j < dataSize; j++)
            {
                double current_distance_squared = 0;
            
                    for (unsigned m = 1; m < featureCount + 1; m++)
                    {
                        if (f && (isElementOf(current_set_of_features, m, size) || m == k))
                        {
                            current_distance_squared += pow(normalizedData[i][m] - normalizedData[j][m], 2);
                    
                        }
                        else if (!f && isElementOf(current_set_of_features, m, size) && m != k)
                        {
                            current_distance_squared += pow(normalizedData[i][m] - normalizedData[j][m], 2);
                    
                        }
                    }
            
                if  (sqrt(current_distance_squared) < minimum_distance && i != j)
                {
                    nearest_neighbor = j;
                    minimum_distance = sqrt(current_distance_squared);
                }//end if
            }//end for
        
        if (fabs(normalizedData[i][0] - normalizedData[nearest_neighbor][0]) < EPSILON) correct_counter++;
        
        
        
        }
        
        
    }//end for
    
    accuracy = static_cast<double>(correct_counter)/static_cast<double>(dataSize - (b - a));
    return accuracy;
    
    // int temp = rand();
    // return temp;
    
}

void displayElements(int* array, int size, bool f)
{
    if (f)
    {
        cout << "{";
        for( unsigned i = 1; i < size; i++)
        {
            cout << array[i] << ", ";
        }
        cout << array[size] << "}";
    }
    else
    {
        int temp[size];
        int counter = 0;
        for ( unsigned i = 1; i < size; i++)
        {
            if (array[i] != 0)
            {
                temp[counter++] = array[i];

            }
        }
        if (counter == 0) 
        {
            cout << "{}";
            return;
        }
        cout << "{";
        cout << temp[0];
        for (unsigned i = 1; i < counter; i++)
        {
            cout << ", " << temp[i];
        }
        cout << "}";
    }
}

int* forwardSearch(int &c, int a, int b, bool f)
{
    int* current_set_of_features = new int[featureCount+1];
    int* best_features;
    double best_accuracy = INT_MIN;
    int counter = 0;
    
    fill(current_set_of_features, current_set_of_features + featureCount+1, 0);
    
    for (unsigned i = 1; i < featureCount + 1; i++)
    {
        if (f) cout << "On the " << i << "th level of the search tree" << endl;
        int feature_to_add_at_this_level;
        double best_so_far_accuracy = INT_MIN;

        
        for (unsigned k = 1; k < featureCount + 1; k++)
        {
            if (!isElementOf(current_set_of_features, k, i-1))
            {
                double accuracy = leave_one_out_cross_validation(current_set_of_features, k, i-1, a, b, 1);
                if (f) cout << "--Considering adding the " << k << "th feature with accuracy of " << accuracy << endl;
                
                
                if (accuracy > best_so_far_accuracy)
                {
                    best_so_far_accuracy = accuracy;
                    feature_to_add_at_this_level = k;
                }//end if
            }//end if
        }//end for
            
        current_set_of_features[i] = feature_to_add_at_this_level;
        if (f) cout << "On level " << i << " I added the feature " << feature_to_add_at_this_level << " to the current set" << endl;
        if (f) cout << "Current set is now ";
        if (f) displayElements(current_set_of_features,i,1);
        if (f) cout << " with an accuracy of " << best_so_far_accuracy*100 << '%' << '.' << endl << endl;
        
        if (best_so_far_accuracy <= best_accuracy) if (f) cout << "***Warning, Accuracy has Decreased. Continuing search in case of local maxima***\n" << endl;
        if (best_so_far_accuracy > best_accuracy)
        {
            best_accuracy = best_so_far_accuracy;
            best_features = current_set_of_features;
            counter = i;
        }
        
    }
    
    cout << "The set of features ";
    displayElements(best_features, counter,1);
    cout << " has an accuracy of " << best_accuracy*100 << "%" << '.' << endl;
    c = counter;
    int* temp = new int[counter+1];
    for (unsigned i = 1; i < counter+1; i++)
    {
        temp[i] = best_features[i];
    }
    return temp;
}

void forwardSearch()
{
    int c;
    forwardSearch(c, 0, 0, 1);
    
}

int* backwardsSearch(int&c, int a, int b, bool f)
{
    int current_set_of_features[featureCount+1];
    int* best_features = new int[featureCount+1];
    double best_accuracy = INT_MIN;
    
    for (unsigned i = 1; i < featureCount + 1; i++) //populates current_set_of_features with all the features
    {
        current_set_of_features[i] = i;
        best_features[i] = i;
    }
    
    
    for (unsigned i = 1; i < featureCount + 1; i++)
    {
        if (f) cout << "On the " << i << "th level of the search tree" << endl;
        int feature_to_remove_at_this_level;
        double best_so_far_accuracy = INT_MIN;
        
        for (unsigned k = 1; k < featureCount + 1; k++)
        {
            if (isElementOf(current_set_of_features, k, featureCount+1))
            {
                double accuracy = leave_one_out_cross_validation(current_set_of_features, k, featureCount, a, b, 0);
                if (f) cout << "--Considering removing the " << k << "th feature with accuracy of " << accuracy << endl;
                
                
                if (accuracy > best_so_far_accuracy)
                {
                    best_so_far_accuracy = accuracy;
                    feature_to_remove_at_this_level = k;
                }//end if
            }//end if
        }//end for
        
        current_set_of_features[feature_to_remove_at_this_level] = 0;
        if (f) cout << "On level " << i << " I removed the feature " << feature_to_remove_at_this_level << " from the current set" << endl;
        if (f) cout << "Current set is now ";
        if (f) displayElements(current_set_of_features,featureCount+1, 0);
        if (f) cout << " with an accuracy of " << best_so_far_accuracy*100 << '%' << '.' << endl << endl;
        
        if (best_so_far_accuracy <= best_accuracy) if (f) cout << "***Warning, Accuracy has Decreased. Continuing search in case of local maxima***\n" << endl;
        if (best_so_far_accuracy > best_accuracy)
        {
            best_accuracy = best_so_far_accuracy;
            for (unsigned i = 1; i < featureCount + 1; i++)
            {
                if (current_set_of_features[i] == 0) best_features[i] = 0;
            }
        }
    }
    
    cout << "The set of features ";
    displayElements(best_features, featureCount+1, 0);
    cout << " has an accuracy of " << best_accuracy*100 << "%" << '.' << endl;
    return best_features;
}

void backwardsSearch()
{
    int c;
    backwardsSearch(c,0,0,1);
}

void customSearch(int n)
{
    int* temp;
    int c = 0;
    int table[featureCount+1];
    fill(table, table + featureCount + 1, 0);
    
    for (unsigned i = 1; i < n+1; i++)
    {
        cout << "---On the " << i << "th iteration of the Forward Selection algorithm" << endl;
        temp = forwardSearch (c, (i-1)*dataSize/n, i*dataSize/n, 0);
        // accuracy = leave_one_out_cross_validation(temp, 0, c, 0, 0, 1);
        for (unsigned j = 1; j < featureCount + 1; j++)
        {
            if (isElementOf(temp, j, c)) table[j]++;
            
        }
        
    }
    
    
    
    cout << endl << endl;
    for (unsigned i = 1; i < n+1; i++)
    {
        cout << "---On the " << i << "th iteration of the Backward Elimination algorithm" << endl;
        temp = backwardsSearch(c, (i-1)*dataSize/n, i*dataSize/n, 0);
        for (unsigned j = 1; j < featureCount + 1; j++)
        {
            if (isElementOf(temp, j, featureCount+1)) table[j]++;
            
        }
    }
    
    cout << "\nOver the " << 2*n << " iterations:" << endl;
    
    for (unsigned i = 1; i < featureCount + 1; i++)
    {
        if (table[i] > 0) cout << "\tFeature " << i << " appeared " << table[i] << " time(s)\n";
    }

    
}


int main()
{
    srand (time(NULL));
    
    string file, input;
    int selection = 0;
    int selection2 = 0;
    string test1 = "CS170BIGtestdata__57.txt";
    string test2 = "CS170Smalltestdata__58.txt";
    
    // int test[] = {0, 2, 3, 4, 5};
    // cout << 5 << " is an element of the array: " << boolalpha << isElementOf(test, 5, 4) << endl;
    
    cout << "Welcome to Phillip's Feature Selection Algorithm!" << endl;
    cout << "Type in the name of the file to test (test1 for \"CS170BIGtestdata__57\", test2 for \"CS170Smalltestdata__58\"): ";
    cin >> input;
    cout << endl;
    
    if (input == "test1") file = test1;
    else if (input == "test2") file = test2;
    else file = input;
    
    readData(file);
    cout << "The data has " << featureCount << " features (not including the class attribute), with " << dataSize << " instances." << endl;
    
    cout << "Please wait while I normalize the data...";
    normalizeData();
    cout << "\tDone!" << endl << endl;
    
    while (selection != 1 && selection != 2 && selection != 3)
    {
    
        cout << "Type the number of the algorithm you want to run." << endl;
        cout << "\t1. Forward Selection" << endl;
        cout << "\t2. Backwards Elimination" << endl;
        cout << "\t3. Phillip's Own Custom Algorithm" << endl;
        cin >> selection;
        cout << endl;
        
        if (selection == 1) forwardSearch();
        else if (selection == 2) backwardsSearch();
        else if (selection == 3) 
        {
            cout << "How many iterations (per algorithm)? ";
            cin >> selection2;
            customSearch(selection2);
        }
        else cout << "Invalid input" << endl;
    }
    

    
    return 0;
    
}