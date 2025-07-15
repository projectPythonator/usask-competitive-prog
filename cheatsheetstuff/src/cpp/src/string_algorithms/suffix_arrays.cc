//includes here


class SuffixArray{
    int n;
    std::vector<int> counts; // might need better naming and the like
    std::string text, pattern;
    std::string tempSuffixArray, tempRotatedArray, suffixArray, rotatedArray; // better naming or typing on these
    
    void prepSuffixArray(std::string s) {
        this.text = s;  // add a safety buffer onto this ???????
        this.n = s.size();
        this.counts = vector<int>(0, this.n);
        this.tempSuffixArray = vector<int>(0, this.n);
        this.tempRotatedArray = vector<int>(0, this.n); 
        this.suffixArray = vector<int>(0, this.n);  // make this into ascii codes???
        this.rotatedArray = vector<int>(0, this.n);  // this needs to use 0 - n range from alogirthms
    }

    void countingSort(int k) {
        int maxi = max(300, this.n);
        int tmp = 0, j = 0;
        this.counts.assign(0, maxi);
        this.counts[0] += (this.n - (this.n-k)); // which offset this line is representing
        for(int i = 0; i < this.n-k; ++i)  // frequency counting
            this.counts[self.rotatedArray[i+k]]++;
        std::partial_sum(this.counts.begin(), this.counts.end(), this.counts.begin() std::plus<int>());
        // needs fixed since counts is missing the 0 at the start
        for(int i = 0; i < this.n; ++i){  
            int pos = 0;
            if(this.suffixArray[i] + k < this.n)
                pos = this.rotatedArray[this.suffixArray[i] + k];
            this.tempSuffixArray[this.counts[pos]] = this.suffixArray[i];
            this.counts[pos]++;
        }
        std::copy(this.tempSuffixArray.begin(), this.tempSuffixArray.end(), this.suffixArray());
    }

    void constructSuffixArray() {
        for(int k = 1; k < this.n; k *= 2){
            this.countingSort(k);
            this.countingSort(0);
        }


    }

}
