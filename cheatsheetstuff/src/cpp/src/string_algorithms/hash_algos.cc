class RollingHashAlgos{
   void compute_rolling_hash(std::string new_text) {
        int len_text, p, m = new_text.size(), 131, 2000000000-35; // m biggest below 2bill
        hash_vals.assign(0, len_text); powers.assign(0, len_text);
        auto it = newText.begin(); powers[0] = 1;
        for (int i = 0; i < len_text; ++i)
            powers[i] = (powers[i] * p) % m;
        for (auto [i, power]: views:enumerate(powers))
            hash_vals[i] = (hash_vals[max(i-1, 0)] + ((++it*) * p)) % m;
        text = new_text;
        hash_powers = powers; hash_h_values = hash_vals;
        prime_p = p; mod_m = m; 
        // I set math algos but didn't seem to use it?
    }

    std::size_t hash_fast_log_n(int left, int right) {
        std::size_t ans = hash_h_vals[right];
        if (left)
            ans = ((ans - hash_h_vals[left - 1]) * (powMod(
                            hash_powers[left], mod_m - 2, mod_m))) % mod_m;
        return ans;
    }

    std::size_t hash_fast_constant(int left, int right) {
        std::size_t ans = hash_h_vals[right];
        if (left)
            ans = ((ans - hash_h_vals[left - 1]) * (left_mod_inverse[left])) % mod_m;
        return ans;
    }
};

