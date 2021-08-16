"""

#include <bits/stdc++.h>

using namespace std;

string ltrim(const string &);
string rtrim(const string &);

/*
 * Complete the 'countHighlyProfitableMonths' function below.
 *
 * The function is expected to return an INTEGER.
 * The function accepts following parameters:
 *  1. INTEGER_ARRAY stockPrices
 *  2. INTEGER k
 */

答案在这个网站上找到：https://stupidtechy.me/threads/highly-profitable-months-hackerrank-c-solutions.105/

int countHighlyProfitableMonths(vector<int> stockPrices, int k) {
    int n=stockPrices.size();

    int count=1;
    vector<int> arr;
    for(int i=0;i+1<n;i++){
        if(stockPrices[i+1]>stockPrices[i])
            count+=1;
        else{
          arr.push_back(count);
          count = 1;
        }
    }

    arr.push_back(count);

    int res = 0;
    for (auto x : arr) {
      if (x >= k)
        res += (x - k + 1);
    }

    return res;
}

int main()


"""