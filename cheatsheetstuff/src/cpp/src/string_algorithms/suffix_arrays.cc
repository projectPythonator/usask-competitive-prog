//includes here


class SuffixArray{
    int n;
    std::vector<int> counts; // might need better naming and the like
    std::string text, pattern;
    std::string tempSuffixArray, tempRankArray, suffixArray, rankArray; // better naming or typing on these
    
    void prepSuffixArray(std::string s) {
        this.text = s;  // add a safety buffer onto this ???????
        this.n = s.size();
        this.tempRankArray = vector<int>(0, this.n); 
        this.suffixArray = vector<int>(0, this.n);  // make this into ascii codes???
        this.rankArray = vector<int>(0, this.n);  // this needs to use 0 - n range from alogirthms
    }

    void countingSort(int k) {
        int maxi = max(300, this.n);
        std::vector<int> counts(maxi, 0);
        std::vector<int> tempSA(this.n);
        for(int i = 0; i < n; ++i)  // frequency counting
            ++counts[(i+k < n)? rankArray[i+k]: 0];
        for(int i = 0, tmp = 0; i < maxi; ++i)
            {int t = counts[i]; counts[i] = tmp; tmp += t;}
        for(int i = 0; i < n; ++i)
            tempSA[counts[
                (suffixArray[i]+k<n)? rankArray[suffixArray[i]+k]: 0]++] = suffixArray[i];
        std::swap(suffixArray, tempSA);
    }

    void constructSuffixArray() {
        int r = 0;
        suffixArray.resize(n);
        rankArray.resize(n);
        itoa(suffixArray.begin(), suffixArray.end(), 0);
        std::copy(text.begin(), text.end(), rankArray.begin());
        for(int k = 1; k < n; k *= 2){
            countingSort(k);
            countingSort(0);
            std::vector<int> tempRA(n);
            tempRA[suffixArray[0]] = r = 0;
            for(int i = 1; i < n; ++i){
                int a = suffixArray[i], b = suffixArray[i-1];
                r += ((rankArray[a] != rankArray[b]) || (rankArray[a+k] != rankArray[b+k]));
                tempRA[a] = r;
            }
            std::swap(rankArray, tempRA);
            if(rankArray[suffixArray[n-1]] == n - 1) break;
        }
    }
}
