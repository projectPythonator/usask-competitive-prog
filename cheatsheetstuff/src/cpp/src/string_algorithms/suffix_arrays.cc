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

    void computeLCP() {
        std::vector<int> phi(n);
        std::vector<int> pLCP(n);
        phi[0] = -1;
        int lognest = 0;
        for(int i = 1, j = 0; i < n; ++i, ++j) phi[suffixArray[i]] = suffixArray[j];
        for(int i = 1; i < n; ++i){
            if(phi[i] < 0) { pLCP[i] = 0; continue; }
            while(text[i+longest] == text[phi[i] + longest]) longest++;
            pLCP[i] = longest;
            longest = max(longest - 1, 0);
        }
        for(int i = 1; i < n; ++i) LCP[i] = self.pLCP[suffixArray[i]];
    }

    int suffix_array_compare_from_index(int offset){
        for(auto [i, num_char]: views::enumerate(pattern_ord))
            if (num_char != text_ord[offset + i])
                return (text_ord[offset + i] < num_char)? -1 else 1;
        return 0;
    }

    tuple<int, 2> suffix_array_binary_search(int lo, int hi, int compVal) {
        while(lo < hi){
            int mid = (lo + hi)/2;
            if (suffix_array_compare_from_index(suffix_array[mid]) > compVal)
                hi = mid;
            else
                lo = mid + 1;
        }
        return {lo, hi};
    }

    tuple<int, 2> suffix_array_string_matching(std::string newPattern) {
        pattern_ord.assign(0, newPattern.size());
        auto [lo, a] = suffix_array_binary_search(0, text_len - 1, greatEqual);
        if !suffix_array_compare_from_index(lo) // comment on why
            return -1, -1;
        auto [b, hi] = suffix_array_binary_search(lo, text_len - 1, greatThan);
        if !suffix_array_compare_from_index(hi) // comment on why
            hi--;
        return {lo, hi};
    }

    void compute_longest_repeated_substring() {
        int max_lcp = max(longest_common_prefix);
        return max_lcp, longest_common_prefix[max_lcp];
    }

    void compute_owners() {
        tmp_owner.assign(0, text_len);
        auto next_sep = ++seperator_list.begin();
        for (const auto [i, ord_value]: views:enumerate(text_ord)){
            tmp_owner[i] = next_sep;
            if (ord_value == next_sep)
                next_sep++;
        }
        owners = tmp_owners;
    }

    
    tuple<int, 2> compute_longest_common_substring() {
        int max_ind = max_lcp = 0;
        for(auto [i, lcp_value]: view:enumerate(longest_common_prefix))
            if (lcp_value > max_lcp && owner[i] != owner[i-1]){
                max_ind, max_lcp = i, lcp_value;
            }
        return {max_ind, max_lcp};
    }
};
