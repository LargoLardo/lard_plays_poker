#pragma once
#include "poker.hpp"
#include <vector>
#include <unordered_map>
#include <limits>
#include <numeric>

// V5's card abstraction is frozen in each checkpoint. It never consumes the
// training RNG and never observes an opponent's cards or an unrevealed board.
namespace v5 {
using namespace poker;
constexpr int action_count = 5, dimensions = 32;
using Features = std::array<float, dimensions>;

inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct Random {
    uint64_t state;
    uint64_t next() { state += 0x9e3779b97f4a7c15ULL; return mix(state); }
    int bounded(int n) {
        const auto size = uint64_t(n), threshold = -size % size;
        uint64_t value; do { value = next(); } while (value < threshold);
        return int(value % size);
    }
};
struct Visible {
    std::array<int, 7> cards{};
    int count = 0;
    uint64_t key = 0;
};
inline Visible canonical(const Cards& cards, int seat, int street) {
    Visible best;
    best.count = 2 + board_size(street);
    best.cards.fill(99);
    std::array<int, 4> permutation{0, 1, 2, 3};
    do {
        std::array<int, 7> candidate{};
        for (int i = 0; i < best.count; ++i) {
            int c = i < 2 ? cards[seat * 2 + i] : cards[i + 2];
            candidate[i] = c / 4 * 4 + permutation[c % 4];
        }
        std::sort(candidate.begin(), candidate.begin() + 2);
        if (street) std::sort(candidate.begin() + 2, candidate.begin() + 5);
        if (candidate < best.cards) best.cards = candidate;
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    best.key = uint64_t(street);
    for (int i = 0; i < best.count; ++i) best.key = (best.key << 6) | uint64_t(best.cards[i]);
    return best;
}
inline int opponent_class(int a, int b) {
    int high = std::max(a / 4, b / 4), low = std::min(a / 4, b / 4);
    bool suited = a % 4 == b % 4;
    if (high == low) return high >= 8 ? 0 : 1;
    if (low >= 8) return suited ? 2 : 3;
    if (high == 12) return suited ? 4 : 5;
    return suited ? 6 : 7;
}
inline Features features(const Visible& visible, int samples, uint64_t seed) {
    Features result{};
    int n = visible.count - 2;
    std::array<bool, 52> known{};
    HandCounts board, hero;
    for (int i = 0; i < visible.count; ++i) {
        known[visible.cards[i]] = true;
        if (i >= 2) board.add(visible.cards[i]);
    }
    hero = board; hero.add(visible.cards[0]); hero.add(visible.cards[1]);
    std::array<int, 52> available{};
    int count = 0;
    for (int c = 0; c < 52; ++c) if (!known[c]) available[count++] = c;
    std::array<double, 8> wins{}, totals{};
    double equity = 0;
    int runouts = n == 5 ? 1 : std::min(32, samples);
    int opponents = std::max(1, samples / runouts);
    Random rng{mix(visible.key ^ seed)};
    for (int run = 0; run < runouts; ++run) {
        auto deck = available;
        HandCounts final_board = board;
        for (int i = 0; i < 5 - n; ++i) {
            int j = i + rng.bounded(count - i);
            std::swap(deck[i], deck[j]); final_board.add(deck[i]);
        }
        auto final_hero = final_board;
        final_hero.add(visible.cards[0]); final_hero.add(visible.cards[1]);
        auto value = evaluate(final_hero);
        double run_wins = 0;
        int compared = 0, used = 5 - n;
        auto compare = [&](int a, int b) {
            auto villain = final_board; villain.add(a); villain.add(b);
            auto other = evaluate(villain);
            double win = value > other ? 1.0 : value == other ? .5 : 0.0;
            int group = opponent_class(a, b);
            wins[group] += win; ++totals[group]; run_wins += win; ++compared;
        };
        if (n == 5) {
            // Seven visible cards leave C(45,2)=990 equally likely hands.
            for (int i = 0; i < count; ++i) for (int j = i + 1; j < count; ++j) compare(deck[i], deck[j]);
        } else for (int i = 0; i < opponents; ++i) {
            int a = used + rng.bounded(count - used);
            int b = used + rng.bounded(count - used - 1);
            if (b >= a) ++b;
            compare(deck[a], deck[b]);
        }
        double strength = run_wins / compared;
        equity += strength / runouts;
        int bin = std::min(15, int(strength * 16));
        for (int i = bin; i < 16; ++i) result[i] += 1.0f / runouts;
    }
    for (int i = 0; i < 8; ++i) result[16 + i] = float((wins[i] + 2 * equity) / (totals[i] + 2));
    result[24] = float(evaluate(hero) >> 20) / 8;
    result[25] = float(std::max(visible.cards[0] / 4, visible.cards[1] / 4)) / 12;
    int suit = int(std::max_element(board.suit_counts.begin(), board.suit_counts.end()) - board.suit_counts.begin());
    int combined = hero.suit_counts[suit];
    result[26] = combined >= 4 && hero.suit_counts[suit] > board.suit_counts[suit] ? 1 : 0;
    int nut = 12;
    while (nut >= 0 && (board.suits[suit] & (1U << (nut + 2)))) --nut;
    result[27] = board.suit_counts[suit] >= 2 && nut >= 0
        && (visible.cards[0] == nut * 4 + suit || visible.cards[1] == nut * 4 + suit) ? 1 : 0;
    result[28] = float(board.suit_counts[suit]) / 5;
    int ranks_count = 0;
    for (int r = 2; r <= 14; ++r) ranks_count += bool(board.mask & (1U << r));
    result[29] = float(n - ranks_count) / 4;
    result[30] = float(std::max(0, highest_rank(board.mask) - 2)) / 12;
    result[31] = float(highest_rank(hero.pairs)) / 14;
    return result;
}
inline double distance(const Features& a, const Features& b) {
    double value = 0;
    // The L1 distance between CDFs is 1-D earth mover distance. Opponent
    // classes and blocker/made-hand features add two equally weighted terms.
    for (int i = 0; i < 16; ++i) value += std::abs(double(a[i]) - b[i]) / 16;
    for (int i = 16; i < dimensions; ++i) { double d = double(a[i]) - b[i]; value += d * d / 8; }
    return value;
}
struct Abstraction {
    int samples = 512;
    uint64_t seed = 1, examples = 0;
    std::array<std::vector<Features>, 3> centers;
    int classify(const Features& values, int street) const {
        const auto& group = centers.at(size_t(street - 1));
        if (group.empty()) throw std::runtime_error("Build a card abstraction before training");
        double best = std::numeric_limits<double>::infinity();
        int chosen = 0;
        for (size_t i = 0; i < group.size(); ++i) {
            double d = distance(values, group[i]);
            if (d < best) { best = d; chosen = int(i); }
        }
        return chosen;
    }
};
struct CardCache {
    std::unordered_map<uint64_t, uint16_t> buckets;
    size_t limit = 0;
    int hand(const Abstraction& abstraction, const Cards& cards, int seat, int street) {
        if (!street) {
            std::mt19937_64 unused(0);
            return card_bucket(cards, seat, 0, 1, unused);
        }
        auto visible = canonical(cards, seat, street);
        auto found = buckets.find(visible.key);
        if (found != buckets.end()) return found->second;
        int value = abstraction.classify(features(visible, abstraction.samples, abstraction.seed), street);
        // Bounded assignment cache; clearing only affects speed, never bucket IDs.
        if (limit) {
            if (buckets.size() >= limit) buckets.clear();
            buckets.emplace(visible.key, uint16_t(value));
        }
        return value;
    }
};
struct Actions {
    int mask = 2;
    std::array<int, action_count> amounts{};
};
inline Actions actions(const State& state) {
    Actions result;
    if (state.terminal()) { result.mask = 0; return result; }
    int high = std::max(state.bets[0], state.bets[1]);
    int maximum = state.stacks[state.actor] + state.bets[state.actor];
    if (state.bets[state.actor] < high) result.mask |= 1;
    auto rounded = [&](double value) { return std::min(maximum, int(std::nearbyint(value))); };
    int pot_after_call = state.pot() + state.to_call();
    result.amounts[2] = rounded(state.street == 0 ? high * (state.history[0].raises ? 3.0 : 2.5) : high + pot_after_call * .5);
    result.amounts[3] = rounded(high + pot_after_call);
    result.amounts[4] = maximum;
    for (int a = 2; a < action_count; ++a) {
        if (a != 4 && state.history[state.street].raises >= 2) continue;
        if (!state.can_raise(result.amounts[a])) continue;
        bool duplicate = false;
        for (int b = 2; b < a; ++b) if ((result.mask & (1 << b)) && result.amounts[b] == result.amounts[a]) duplicate = true;
        if (!duplicate) result.mask |= 1 << a;
    }
    return result;
}
struct Position {
    State state;
    uint64_t history = 1;
    void act(int action) {
        auto choices = actions(state);
        if (!(choices.mask & (1 << action))) throw std::runtime_error("Illegal V5 action");
        if (history >> 61) throw std::runtime_error("Betting history exceeds V5 encoding");
        history = (history << 3) | uint64_t(action + 1);
        state.act(action >= 2 ? 2 : action, choices.amounts[action]);
    }
    int observe(int kind, double ratio, bool jam) {
        if (state.terminal()) return -1;
        auto choices = actions(state);
        int selected = kind;
        if (kind >= 2) {
            double best = std::numeric_limits<double>::infinity();
            selected = -1;
            int high = std::max(state.bets[0], state.bets[1]);
            for (int a = 2; a < action_count; ++a) if (choices.mask & (1 << a)) {
                double candidate = double(choices.amounts[a] - high) / (state.pot() + state.to_call());
                double error = jam ? (a == 4 ? 0 : 1e10) : std::abs(candidate - ratio);
                if (error < best) { best = error; selected = a; }
            }
        }
        if (selected < 0 || !(choices.mask & (1 << selected))) return -1;
        act(selected); return selected;
    }
};
inline Position replay(uint64_t history) {
    std::array<int, 21> sequence{};
    int count = 0;
    while (history != 1) {
        if (!history || count == int(sequence.size()) || (history & 7) < 1 || (history & 7) > 5)
            throw std::runtime_error("Invalid V5 history");
        sequence[count++] = int(history & 7) - 1; history >>= 3;
    }
    Position result;
    while (count) result.act(sequence[--count]);
    return result;
}
struct Key {
    uint64_t history = 0;
    uint32_t context = 0;
    bool operator==(const Key& other) const { return history == other.history && context == other.context; }
};
struct Hash { size_t operator()(const Key& key) const { return size_t(mix(key.history ^ (uint64_t(key.context) << 32))); } };
inline Key key(const Position& position, int hand) {
    return {position.history, uint32_t(hand) | (uint32_t(position.state.actor) << 16)
        | (uint32_t(position.state.street) << 17) | (uint32_t(actions(position.state).mask) << 19)};
}
struct Node {
    std::array<double, action_count> regret{}, strategy{};
    uint64_t visits = 0;
    void add(const Node& other) {
        for (int a = 0; a < action_count; ++a) { regret[a] += other.regret[a]; strategy[a] += other.strategy[a]; }
        visits += other.visits;
    }
};
struct Entry { Key key; Node node; };
class Nodes {
    std::vector<Entry> entries;
    size_t used = 0, maximum = 1024;
    size_t slot(const Key& key) const {
        size_t index = Hash{}(key) & (entries.size() - 1);
        while (entries[index].key.history && !(entries[index].key == key)) index = (index + 1) & (entries.size() - 1);
        return index;
    }
public:
    explicit Nodes(size_t bytes) {
        while (maximum <= bytes / sizeof(Entry) / 3) maximum *= 2;
        if (maximum * sizeof(Entry) * 3 / 2 > bytes) maximum /= 2;
        if (maximum < 1024) throw std::runtime_error("V5 memory budget too small");
        entries.resize(1024);
    }
    size_t size() const { return used; }
    size_t bytes() const { return entries.size() * sizeof(Entry); }
    size_t limit() const { return maximum * 7 / 10; }
    const Node* get(const Key& key) const { const auto& entry = entries[slot(key)]; return entry.key.history ? &entry.node : nullptr; }
    bool prepare(size_t additional) {
        size_t target = entries.size();
        while (used + additional > target * 7 / 10 && target < maximum) target *= 2;
        if (used + additional > target * 7 / 10) return false;
        if (target != entries.size()) {
            std::vector<Entry> replacement(target);
            auto old = std::move(entries); entries = std::move(replacement);
            for (const auto& entry : old) if (entry.key.history) entries[slot(entry.key)] = entry;
        }
        return true;
    }
    void add(const Key& key, const Node& node) {
        auto& entry = entries[slot(key)];
        if (!entry.key.history) { entry.key = key; ++used; }
        entry.node.add(node);
    }
    template<class F> void each(F callback) const { for (const auto& entry : entries) if (entry.key.history) callback(entry.key, entry.node); }
    void discount(double factor) {
        for (auto& entry : entries) if (entry.key.history) for (int a = 0; a < action_count; ++a) {
            entry.node.regret[a] *= factor; entry.node.strategy[a] *= factor;
        }
    }
};
} // namespace v5
