#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace poker {
using Cards = std::array<int, 9>; // BB hole, SB hole, five board cards.
constexpr const char* ranks = "23456789TJQKA";
constexpr const char* suits = "shdc";
constexpr const char* histories[] = {"root", "limped", "vs_open", "vs_3bet", "vs_4bet"};
constexpr const char* sizes[] = {"Limp", "~2.0bb raise", "~2.75bb raise", "~6.0bb raise", "~10.0bb raise", "~25.0bb raise", "Jam (>25.00bb) raise"};
constexpr const char* pot_sizes[] = {"small", "medium", "large", "overbet"};
constexpr const char* sprs[] = {"short", "mid", "mid_deep", "deep"};

inline int card(const std::string& text) {
    if (text.size() != 2) throw std::runtime_error("Invalid card: " + text);
    auto r = std::string(ranks).find(text[0]);
    auto s = std::string(suits).find(text[1]);
    if (r == std::string::npos || s == std::string::npos) throw std::runtime_error("Invalid card: " + text);
    return int(r * 4 + s);
}

inline int straight(uint32_t mask) {
    // All 13-rank masks fit in an 8 KiB table; initialize once per process.
    static const auto table = [] {
        std::array<uint8_t, 8192> result{};
        for (uint32_t bits = 0; bits < result.size(); ++bits) {
            for (int high = 14; high >= 6; --high)
                if (((bits >> (high - 6)) & 31U) == 31U) { result[bits] = uint8_t(high); break; }
            if (!result[bits] && (bits & 4111U) == 4111U) result[bits] = 5;
        }
        return result;
    }();
    return table[(mask >> 2) & 8191U];
}

struct HandCounts {
    std::array<uint32_t, 4> suits{};
    std::array<uint8_t, 4> suit_counts{};
    uint32_t mask = 0, pairs = 0, trips = 0, quads = 0;
    void add(int card) {
        int r = card / 4 + 2, s = card % 4;
        uint32_t bit = 1U << r;
        // Multiplicity masks contain ranks seen at least 2/3/4 times.
        quads |= trips & bit; trips |= pairs & bit; pairs |= mask & bit;
        mask |= bit; suits[s] |= bit; ++suit_counts[s];
    }
};

inline uint32_t score(int category, const std::array<int, 5>& values) {
    uint32_t result = uint32_t(category);
    for (int value : values) result = (result << 4) | uint32_t(value);
    return result;
}

inline int highest_rank(uint32_t mask) {
#ifdef _MSC_VER
    unsigned long rank = 0;
    return _BitScanReverse(&rank, mask) ? int(rank) : 0;
#else
    return mask ? 31 - __builtin_clz(mask) : 0;
#endif
}

inline std::array<int, 5> high_cards(uint32_t mask) {
    std::array<int, 5> values{};
    for (int i = 0; i < 5 && mask; ++i) {
        values[i] = highest_rank(mask);
        mask &= ~(1U << values[i]);
    }
    return values;
}

// Evaluate 5-7 cards directly from rank/suit bitmasks. Higher wins.
inline uint32_t evaluate(const HandCounts& hand) {
    const auto& suit_masks = hand.suits;
    const auto& suit_counts = hand.suit_counts;
    uint32_t mask = hand.mask;
    int flush = -1;
    for (int s = 0; s < 4; ++s) if (suit_counts[s] >= 5) {
        int high = straight(suit_masks[s]);
        if (high) return score(8, {high, 0, 0, 0, 0});
        flush = s;
    }
    int quad = highest_rank(hand.quads), trip = highest_rank(hand.trips);
    if (quad) return score(7, {quad, highest_rank(mask & ~(1U << quad)), 0, 0, 0});
    int full_pair = highest_rank(hand.pairs & ~(1U << trip));
    if (trip && full_pair) return score(6, {trip, full_pair, 0, 0, 0});
    if (flush >= 0) return score(5, high_cards(suit_masks[flush]));
    int high = straight(mask);
    if (high) return score(4, {high, 0, 0, 0, 0});
    if (trip) { auto k = high_cards(mask & ~(1U << trip)); return score(3, {trip, k[0], k[1], 0, 0}); }
    int pair = highest_rank(hand.pairs), second_pair = highest_rank(hand.pairs & ~(1U << pair));
    if (second_pair) return score(2, {pair, second_pair, highest_rank(mask & ~((1U << pair) | (1U << second_pair))), 0, 0});
    if (pair) { auto k = high_cards(mask & ~(1U << pair)); return score(1, {pair, k[0], k[1], k[2], 0}); }
    return score(0, high_cards(mask));
}

inline uint32_t evaluate(const int* cards, int count) {
    HandCounts hand;
    for (int i = 0; i < count; ++i) hand.add(cards[i]);
    return evaluate(hand);
}

struct History {
    uint8_t length = 0, raises = 0, first = 0, second = 0, third = 0;
    uint8_t raised_by = 0;
    void add(int action, int actor) {
        if (length == 0) first = uint8_t(action);
        if (length == 1) second = uint8_t(action);
        if (length == 2) third = uint8_t(action);
        ++length;
        if (action == 2) { ++raises; raised_by |= uint8_t(1 << actor); }
    }
    int bucket() const {
        if (!length) return 0;
        if (length == 1) return first == 1 ? 1 : 2;
        if (length == 2) return first == 2 && second == 2 ? 3 : 2;
        if (length == 3) return first == 2 && second == 2 && third == 2 ? 4 : 3;
        return 4;
    }
};

// Chips use half-big-blind units: 200 = 100bb. No heap-allocated game tree.
struct State {
    int actor = 1, street = 0;
    std::array<int, 2> stacks{198, 199}, bets{2, 1}, invested{2, 1};
    std::array<History, 4> history{};
    int pending = 3, min_raise = 2, folded = -1;
    int pot() const { return invested[0] + invested[1]; }
    int to_call() const { return std::abs(bets[0] - bets[1]); }
    bool terminal() const { return street == 4; }
    bool can_raise(int amount) const {
        if (terminal() || stacks[1 - actor] == 0) return false;
        int high = std::max(bets[0], bets[1]);
        int maximum = stacks[actor] + bets[actor];
        int effective = std::min(maximum, stacks[1 - actor] + bets[1 - actor]);
        int minimum = std::min(effective, high + min_raise);
        return maximum > high && amount > high && amount >= minimum && amount <= maximum;
    }
    int raise_size(bool preflop_only, std::mt19937_64& rng) const {
        double amount;
        if (preflop_only) {
            constexpr double fractions[] = {1.0 / 3, .5, 2.0 / 3, 1};
            amount = std::max(bets[0], bets[1]) + pot() * fractions[std::uniform_int_distribution<int>(0, 3)(rng)];
        } else if (street == 0) amount = std::max(bets[0], bets[1]) * 3;
        else amount = std::max(bets[0], bets[1]) + pot() * .5;
        // Python round() rounds half big blinds to the nearest even integer.
        int rounded = int(std::nearbyint(amount / 2)) * 2;
        int maximum = stacks[actor] + bets[actor];
        if (street != 1 && history[street].bucket() == 4) rounded = maximum;
        return std::min(rounded, maximum);
    }
    void act(int action, int amount = 0) {
        if (terminal()) throw std::runtime_error("Action after hand ended");
        int player = actor, other = 1 - player;
        int high = std::max(bets[0], bets[1]);
        if (action == 0) {
            if (bets[player] >= high) throw std::runtime_error("Fold when checking is free");
            history[street].add(action, player); folded = player; street = 4; return;
        }
        if (action == 2 && !can_raise(amount)) throw std::runtime_error("Illegal raise");
        int chips = action == 2 ? amount - bets[player] : std::min(stacks[player], high - bets[player]);
        stacks[player] -= chips; bets[player] += chips; invested[player] += chips;
        history[street].add(action, player);
        pending &= ~(1 << player);
        if (action == 2) {
            min_raise = std::max(min_raise, amount - high);
            pending = 1 << other;
        }
        if (!pending || (bets[0] == bets[1] && (!stacks[0] || !stacks[1]))) {
            ++street;
            if (street == 4 || !stacks[0] || !stacks[1]) { street = 4; return; }
            bets = {0, 0}; actor = 0; pending = 3; min_raise = 2;
        } else actor = other;
    }
    double payoff(int winner) const {
        if (folded == 0) return -invested[0] / 2.0;
        if (folded == 1) return invested[1] / 2.0;
        int matched = std::min(invested[0], invested[1]);
        return winner == 2 ? 0 : (winner == 0 ? matched : -matched) / 2.0;
    }
};

inline int board_size(int street) { return street == 0 ? 0 : street + 2; }
inline int max_suit(const Cards& cards, int n) {
    std::array<int, 4> counts{};
    for (int i = 0; i < n; ++i) ++counts[cards[4 + i] % 4];
    return *std::max_element(counts.begin(), counts.end());
}
inline bool paired(const Cards& cards, int n) {
    uint32_t seen = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t bit = 1U << (cards[4 + i] / 4);
        if (seen & bit) return true;
        seen |= bit;
    }
    return false;
}
inline int straight_draw(const Cards& cards, int n) {
    uint32_t mask = 0;
    for (int i = 0; i < n; ++i) mask |= 1U << (cards[4 + i] / 4);
    int best = 0;
    for (int low = 0; low < 9; ++low) {
        int count = 0;
        for (int r = low; r < low + 5; ++r) count += bool(mask & (1U << r));
        best = std::max(best, count);
    }
    if (mask & (1U << 12)) {
        int count = 1;
        for (int r = 0; r < 4; ++r) count += bool(mask & (1U << r));
        best = std::max(best, count);
    }
    return best;
}

inline int card_bucket(const Cards& cards, int actor, int street, int samples, std::mt19937_64& rng) {
    if (!street) {
        int a = cards[actor * 2] / 4, b = cards[actor * 2 + 1] / 4;
        if (ranks[a] > ranks[b]) std::swap(a, b); // Original buckets sort rank characters.
        return (a * 13 + b) * 2 + (cards[actor * 2] % 4 == cards[actor * 2 + 1] % 4);
    }
    int n = board_size(street);
    std::array<bool, 52> known{};
    known[cards[actor * 2]] = known[cards[actor * 2 + 1]] = true;
    for (int i = 0; i < n; ++i) known[cards[4 + i]] = true;
    std::array<int, 52> deck{};
    int available = 0;
    for (int c = 0; c < 52; ++c) if (!known[c]) deck[available++] = c;
    HandCounts visible;
    for (int i = 0; i < n; ++i) visible.add(cards[4 + i]);
    auto hero_visible = visible;
    hero_visible.add(cards[actor * 2]); hero_visible.add(cards[actor * 2 + 1]);
    uint32_t hero_now = evaluate(hero_visible);
    int wins2 = 0, ahead = 0, behind = 0, improve = 0, worsen = 0;
    // Partial Fisher-Yates samples without replacement, restored after each sample.
    std::array<int, 4> swaps{};
    for (int sample = 0; sample < samples; ++sample) {
        int draws = 7 - n;
        for (int i = 0; i < draws; ++i) {
            int j = std::uniform_int_distribution<int>(i, available - 1)(rng);
            swaps[i] = j; std::swap(deck[i], deck[j]);
        }
        auto runout = visible;
        for (int i = n; i < 5; ++i) runout.add(deck[2 + i - n]);
        auto hero = runout, villain = runout;
        hero.add(cards[actor * 2]); hero.add(cards[actor * 2 + 1]);
        villain.add(deck[0]); villain.add(deck[1]);
        auto h = street == 3 ? hero_now : evaluate(hero), v = evaluate(villain);
        wins2 += h > v ? 2 : h == v ? 1 : 0;
        if (street < 3) {
            auto villain_visible = visible;
            villain_visible.add(deck[0]); villain_visible.add(deck[1]);
            auto v_now = evaluate(villain_visible);
            if (hero_now > v_now) { ++ahead; worsen += h < v; }
            if (hero_now < v_now) { ++behind; improve += h > v; }
        }
        for (int i = draws - 1; i >= 0; --i) std::swap(deck[i], deck[swaps[i]]);
    }
    int equity = std::min(wins2 * 4 / samples, 7);
    bool flush_done = max_suit(cards, n) >= 3 && max_suit(cards, n - 1) < 3;
    bool straight_done = straight_draw(cards, n) >= 3 && straight_draw(cards, n - 1) < 3;
    if (street == 3) return ((equity * 2 + flush_done) * 2 + straight_done) * 2 + paired(cards, n);
    int positive = behind ? std::min(improve * 4 / behind, 3) : 0;
    int negative = ahead ? std::min(worsen * 4 / ahead, 3) : 0;
    int texture = max_suit(cards, n) >= 3 ? 0 : max_suit(cards, n) == 2 ? 1 : 2;
    int result = (((equity * 4 + positive) * 4 + negative) * 3 + texture) * 2 + paired(cards, n);
    if (street == 2) result = (result * 2 + flush_done) * 2 + straight_done;
    return result;
}

inline uint32_t bucket(const State& state, int hand) {
    int stack = 0, size = 0, spr = 0, history = state.history[state.street].bucket(), previous = 0;
    if (state.street == 0) {
        int effective = std::min(state.stacks[0], state.stacks[1]);
        stack = effective < 40 ? 0 : effective < 100 ? 1 : 2;
        double call = state.to_call() / 2.0;
        if (history == 1 || !call) size = 0;
        else if (call <= 2) size = 1;
        else if (call <= 2.75) size = 2;
        else if (call < 6) size = 3;
        else if (call < 10) size = 4;
        else if (call < 25) size = 5;
        else size = 6;
    } else {
        double ratio = double(state.to_call()) / state.pot();
        size = ratio < .4 ? 0 : ratio < .75 ? 1 : ratio < 1.1 ? 2 : 3;
        double value = double(state.stacks[state.actor]) / state.pot();
        spr = value > 10 ? 3 : value > 4 ? 2 : value > 1.5 ? 1 : 0;
        if (state.street == 1) history = std::min(int(state.history[1].raises), 3);
        else previous = bool(state.history[state.street - 1].raised_by & (1 << state.actor));
    }
    return uint32_t(hand) | (uint32_t(state.actor) << 16) | (uint32_t(state.street) << 17)
        | (uint32_t(history) << 19) | (uint32_t(size) << 22) | (uint32_t(spr) << 25)
        | (uint32_t(stack) << 27) | (uint32_t(previous) << 29);
}

inline int legal_mask(const State& state, int amount) {
    return 2 | (state.bets[state.actor] < std::max(state.bets[0], state.bets[1]) ? 1 : 0)
        | (state.can_raise(amount) ? 4 : 0);
}

inline uint32_t bucket(const State& state, int hand, int mask) {
    // Two spare key bits distinguish fold/raise availability; call is always legal.
    return bucket(state, hand) | (uint32_t(mask & 1) << 30) | (uint32_t((mask >> 2) & 1) << 31);
}

inline int key_actions(uint32_t key) {
    return 2 | int((key >> 30) & 1) | (int((key >> 31) & 1) << 2);
}

inline std::string hand_json(int hand, int street) {
    if (!street) {
        bool suited = hand % 2; hand /= 2;
        return std::string("\"") + ranks[hand / 13] + ranks[hand % 13] + (suited ? "s\"" : "o\"");
    }
    std::array<int, 8> parts{};
    int count = street == 1 ? 5 : street == 2 ? 7 : 4;
    constexpr const char* textures[] = {"monotone", "two_tone", "rainbow"};
    if (street == 3) {
        parts[3] = hand % 2; hand /= 2; parts[2] = hand % 2; hand /= 2;
        parts[1] = hand % 2; parts[0] = hand / 2;
    } else {
        if (street == 2) { parts[6] = hand % 2; hand /= 2; parts[5] = hand % 2; hand /= 2; }
        parts[4] = hand % 2; hand /= 2; parts[3] = hand % 3; hand /= 3;
        parts[2] = hand % 4; hand /= 4; parts[1] = hand % 4; parts[0] = hand / 4;
    }
    std::string out = "[";
    for (int i = 0; i < count; ++i) {
        if (i) out += ',';
        if (street != 3 && i == 3) out += std::string("\"") + textures[parts[i]] + '"';
        else if ((street == 3 && i > 0) || i >= 4) out += parts[i] ? "true" : "false";
        else out += std::to_string(parts[i]);
    }
    return out + ']';
}

inline std::string bucket_json(uint32_t key, bool web_key = false, bool with_actions = false) {
    int hand = key & 65535, actor = (key >> 16) & 1, street = (key >> 17) & 3;
    int history = (key >> 19) & 7, size = (key >> 22) & 7;
    int spr = (key >> 25) & 3, stack = (key >> 27) & 3, previous = (key >> 29) & 1;
    std::string position = actor ? "SB" : "BB";
    constexpr const char* stacks[] = {"short", "medium", "deep"};
    if (!street && web_key) {
        auto h = hand_json(hand, street);
        return h.substr(1, h.size() - 2) + '|' + position + '|' + stacks[stack] + '|' + histories[history] + '|' + sizes[size]
            + (with_actions ? "|" + std::to_string(key_actions(key)) : "");
    }
    std::string out = "[" + hand_json(hand, street) + ",\"" + position + "\",";
    if (!street) return out + '"' + stacks[stack] + "\",\"" + histories[history] + "\",\"" + sizes[size] + '"'
        + (with_actions ? "," + std::to_string(key_actions(key)) : "") + ']';
    out += street == 1 ? std::to_string(history) : std::string("\"") + histories[history] + '"';
    out += std::string(",\"") + pot_sizes[size] + "\",\"" + sprs[spr] + '"';
    if (street > 1) out += previous ? ",true" : ",false";
    return out + (with_actions ? "," + std::to_string(key_actions(key)) : "") + ']';
}
} // namespace poker
