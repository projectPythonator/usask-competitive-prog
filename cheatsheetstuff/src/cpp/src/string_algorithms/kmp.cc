// might need a few comments and few fixing up to do on it

#include <string>

class KMP_string_search {
    int n, m; // better place 
    std::string_view text, pattern;
    std::vector<int> back_lookup; // type on this needs fixed
    void prepKMP(std::string s){
        this.n = s.size();
        this.text = s;  // use string_view for proper use
    }

    void preprocess(std::string new_pat) {
        this.m = new_pat.size();
        this.pattern = new_pat;
        this.back_lookup = std::vector<int>(0, this.m + 1); // string with all 0s of size m+1
        int j = this.back_lookup[0] = -1;
        for(int i = 0; i < this.m; ++i){
            while(j >= 0 && this.pattern[i] != this.pattern[j])
                j = this.back_lookup[j];
            j++;
            this.back_lookup[i+1] = j;
        }}

    std::vector<int> kmpSearch(){
        std::vector<int> r_starts;  // maybe take this in as a parameter maybe???
        int j = 0;
        for(int i = 0; i < this.n; ++i){
            while(j >= 0 && this.text[i] != this.pattern[j])
                j = this.back_lookup[j];
            j++;
            if(j == this.m){
                r_starts.push_back(1+i-j);  // needs to explain why this equation
                j = this.back_lookup[j];
            }}
        return r_starts;
    }
}
