#define MODEL_V5_TESTING
#include "train_v5.cpp"
#include <cassert>
#include <set>

int main(int argc, char** argv) {
    if (argc > 1 && (std::string(argv[1]) == "states" || std::string(argv[1]) == "tree")) {
        auto print = [](const v5::Position& position) {
            const auto& s = position.state;
            auto choices = v5::actions(s);
            std::cout << "{\"history\":\"" << position.history << "\",\"mask\":" << choices.mask << ",\"amounts\":[";
            for (int a = 0; a < 5; ++a) { if (a) std::cout << ','; std::cout << choices.amounts[a] / 2.0; }
            std::cout << "],\"game\":{\"street\":" << s.street << ",\"agent\":" << s.actor
                << ",\"stacks\":[" << s.stacks[0] / 2.0 << ',' << s.stacks[1] / 2.0
                << "],\"bets\":[" << s.bets[0] / 2.0 << ',' << s.bets[1] / 2.0
                << "],\"pot\":" << (s.pot() - s.bets[0] - s.bets[1]) / 2.0
                << ",\"lastRaise\":" << s.min_raise / 2.0 << ",\"histories\":[";
            for (int street = 0; street < 4; ++street) {
                if (street) std::cout << ',';
                std::cout << '[';
                for (int i = 0; i < s.history[street].raises; ++i) { if (i) std::cout << ','; std::cout << "\"raise\""; }
                std::cout << ']';
            }
            std::cout << "]}}\n";
        };
        if (std::string(argv[1]) == "tree") {
            auto walk = [&](auto&& self, v5::Position position) -> void {
                if (position.state.terminal()) return;
                print(position);
                auto choices = v5::actions(position.state);
                for (int a = 0; a < 5; ++a) if (choices.mask & (1 << a)) {
                    auto next = position; next.act(a); self(self, next);
                }
            };
            walk(walk, v5::Position{});
        } else {
            std::string line;
            while (std::getline(std::cin, line)) {
                std::istringstream input(line); v5::Position position; int action;
                while (input >> action) position.act(action);
                print(position);
            }
        }
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "features") {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream input(line); Cards cards{};
            for (int& c : cards) { std::string token; input >> token; c = card(token); }
            int seat, street, samples; input >> seat >> street >> samples;
            auto visible = v5::canonical(cards, seat, street);
            auto values = v5::features(visible, samples, 1);
            std::cout << std::setprecision(17) << "{\"canonical\":\"" << visible.key << "\",\"features\":[";
            for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
            std::cout << "]}\n";
        }
        return 0;
    }
    auto assets = v5::learn(32, 7, 32, {4, 4, 4}, 2, 2);
    Cards cards{card("As"), card("Js"), card("Kd"), card("Kc"), card("Qs"), card("8s"), card("2h"), card("5d"), card("Tc")};
    for (int street = 1; street < 4; ++street) {
        auto base = v5::canonical(cards, 0, street);
        auto values = v5::features(base, 32, 7);
        assert(values == v5::features(base, 32, 7));
        std::array<int, 4> permutation{0, 1, 2, 3};
        do {
            auto renamed = cards;
            for (int& c : renamed) c = c / 4 * 4 + permutation[c % 4];
            std::swap(renamed[0], renamed[1]); std::swap(renamed[4], renamed[6]);
            auto other = v5::canonical(renamed, 0, street);
            assert(base.key == other.key && values == v5::features(other, 32, 7));
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        auto hidden = cards; std::swap(hidden[2], hidden[3]);
        if (street < 3) std::swap(hidden[8], hidden[3]);
        assert(base.key == v5::canonical(hidden, 0, street).key);
        v5::CardCache cache; cache.limit = 1;
        assert(cache.hand(assets, cards, 0, street) == cache.hand(assets, hidden, 0, street));
    }
    std::unordered_map<v5::Key, int, v5::Hash> contexts;
    uint64_t histories = 0;
    auto walk = [&](auto&& self, v5::Position position) -> void {
        if (position.state.terminal()) return;
        ++histories;
        auto choices = v5::actions(position.state);
        assert(choices.mask & 2);
        assert(contexts.emplace(v5::key(position, 0), choices.mask).second);
        for (int a = 0; a < 5; ++a) if (choices.mask & (1 << a)) {
            if (a >= 2) { assert(position.state.can_raise(choices.amounts[a])); for (int b = 2; b < a; ++b) if (choices.mask & (1 << b)) assert(choices.amounts[a] != choices.amounts[b]); }
            auto next = position; next.act(a);
            auto replay = v5::replay(next.history);
            assert(replay.history == next.history && replay.state.street == next.state.street && replay.state.bets == next.state.bets);
            self(self, next);
        }
    };
    walk(walk, v5::Position{});
    assert(histories == 13608);
    auto root = v5::actions(State{});
    assert(root.amounts[2] == 5 && root.amounts[3] == 6 && root.amounts[4] == 200 && root.mask == 31);

    v5::Model model(64 * 1048576, 0); model.abstraction = assets; model.discount_every = 10;
    v5::Node value; value.regret[0] = -4; value.regret[1] = 6; value.strategy[1] = 8; value.visits = 2;
    std::unordered_map<v5::Key, v5::Node, v5::Hash> changes{{v5::key(v5::Position{}, 0), value}};
    model.merge(changes, 1, 10);
    auto node = model.nodes.get(changes.begin()->first);
    assert(node->regret[0] == -2 && node->regret[1] == 3 && node->strategy[1] == 4 && node->visits == 2 && model.periods == 1);
    std::unordered_map<v5::Key, v5::Node, v5::Hash> empty;
    model.merge(empty, 1, 10); node = model.nodes.get(changes.begin()->first);
    assert(std::abs(node->strategy[1] - 8.0 / 3) < 1e-12 && node->visits == 2);

    v5::Model parallel(64 * 1048576, 1048576); parallel.abstraction = assets; parallel.workers = 3; parallel.chunk = 8; parallel.discount_every = 100;
    v5::Model same(64 * 1048576, 0); same.abstraction = assets; same.workers = 3; same.chunk = 8; same.discount_every = 100;
    v5::Pool first(parallel), second(same);
    assert(first.step(24) && second.step(24));
    auto saved = fs::temp_directory_path() / ("lard-v5-test-" + std::to_string(uint64_t(Clock::now().time_since_epoch().count())) + ".bin");
    v5::save(parallel, saved);
    v5::Model resumed(64 * 1048576, 0); v5::load(resumed, saved);
    v5::Pool third(resumed);
    assert(first.step(24) && second.step(24) && third.step(24));
    assert(parallel.rng == same.rng && parallel.rng == resumed.rng && parallel.touches == resumed.touches);
    parallel.nodes.each([&](const v5::Key& key, const v5::Node& node) {
        const auto* a = same.nodes.get(key); const auto* b = resumed.nodes.get(key);
        assert(a && b && node.regret == a->regret && node.regret == b->regret && node.strategy == a->strategy && node.strategy == b->strategy && node.visits == b->visits);
    });
    v5::Model limited(64 * 1048576, 0, 1); limited.abstraction = assets; limited.workers = 3; limited.chunk = 8;
    v5::Pool small(limited); auto prior = limited.rng;
    assert(!small.step(24) && limited.rng == prior && !limited.iterations && !limited.touches && !limited.nodes.size());
    fs::remove(saved);
    std::cout << "V5 deterministic cards, exact histories (" << histories << "), action masks, discounting, parallel resume and rollback checks passed\n";
}
