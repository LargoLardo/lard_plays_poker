#define POKER_TESTING
#include "trainer.cpp"
#include <cassert>

void unit_tests() {
    assert(sizeof(Entry) <= 64);
    State state;
    assert(state.actor == 1 && state.to_call() == 1);
    state.act(1); assert(state.actor == 0 && state.street == 0);
    state.act(1); assert(state.street == 1 && state.actor == 0 && state.pot() == 4);
    state.act(2, 2); state.act(2, 198); state.act(1);
    assert(state.terminal() && state.payoff(0) == 100 && state.payoff(1) == -100);
    State fold; fold.act(0); assert(fold.payoff(0) == .5);
    State jam; jam.act(2, 200); jam.act(1); assert(jam.terminal());
    std::mt19937_64 rng(1);
    State raises;
    assert(raises.raise_size(false, rng) == 6);
    raises.act(2, 6); raises.act(2, 18); raises.act(2, 54);
    assert(raises.raise_size(false, rng) == 200);

    constexpr size_t budget = 4 * 1024 * 1024;
    Trainer a(false, 10, 9, budget, 0), b(false, 10, 9, budget, 0);
    for (int i = 0; i < 30; ++i) { assert(a.step()); assert(b.step()); }
    auto path = fs::temp_directory_path() / ("lard-cpp-test-" + std::to_string(Clock::now().time_since_epoch().count()) + ".bin");
    save(b, path);
    Trainer resumed(true, 100, 1, budget, 0); load(resumed, path, false, false);
    assert(!resumed.preflop && resumed.samples == 10 && resumed.iterations == 30);
    for (int i = 0; i < 30; ++i) { assert(a.step()); assert(resumed.step()); }
    assert(a.rng == resumed.rng && a.iterations == resumed.iterations && a.nodes.size() == resumed.nodes.size());
    a.nodes.each([&](uint32_t key, const Node& node) {
        const auto* other = resumed.nodes.get(key); assert(other);
        assert(node.regret == other->regret && node.strategy == other->strategy && node.visits == other->visits);
    });
    fs::remove(path);
    Trainer limited(false, 10, 9, budget, 1);
    auto before = limited.rng;
    assert(!limited.step() && limited.iterations == 0 && limited.nodes.size() == 0 && limited.rng == before);
    Trainer parallel(false, 10, 9, budget, 0), parallel_copy(false, 10, 9, budget, 0);
    parallel.workers = parallel_copy.workers = 3;
    parallel.chunk_size = parallel_copy.chunk_size = 4;
    ParallelTrainer pool(parallel), pool_copy(parallel_copy);
    for (int i = 0; i < 4; ++i) { assert(pool.step(12)); assert(pool_copy.step(12)); }
    assert(parallel.rng == parallel_copy.rng && parallel.iterations == 48);
    parallel.nodes.each([&](uint32_t key, const Node& node) {
        const auto* other = parallel_copy.nodes.get(key); assert(other);
        assert(node.regret == other->regret && node.strategy == other->strategy && node.visits == other->visits);
        for (int action = 0; action < 3; ++action) if (!(key_actions(key) & (1 << action)))
            assert(node.regret[action] == 0 && node.strategy[action] == 0);
    });
    Trainer parallel_limited(false, 10, 9, budget, 1);
    parallel_limited.workers = 3; parallel_limited.chunk_size = 4;
    auto parallel_rng = parallel_limited.rng;
    ParallelTrainer limited_pool(parallel_limited);
    assert(!limited_pool.step(12) && parallel_limited.iterations == 0 && parallel_limited.nodes.size() == 0 && parallel_limited.rng == parallel_rng);
    Trainer average(false, 1, 1, budget, 0);
    average.cards = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (auto& row : average.hands) row.fill(-1);
    auto root = bucket(State{}, card_bucket(average.cards, 1, 0, 1, average.rng), 7);
    Node base; base.regret[0] = 100;
    average.nodes.add(root, base);
    assert(average.traverse(State{}, 0) == .5);
    assert(average.delta.at(root).strategy[0] == 1 && average.delta.at(root).visits == 1);
    assert((average.delta.at(root).regret == std::array<double, 3>{}));
    average.delta.clear();
    average.traverse(State{}, 1);
    assert((average.delta.at(root).strategy == std::array<double, 3>{}) && average.delta.at(root).visits == 0);
    Nodes table(budget);
    assert(table.prepare(1000));
    for (uint32_t i = 0; i < 1000; ++i) { Node n; n.visits = i; table.add(i, n); }
    assert(table.prepare(5000));
    for (uint32_t i = 0; i < 1000; ++i) assert(table.get(i)->visits == i);
    assert(!table.prepare(1'000'000));

    std::unordered_map<uint32_t, int> masks, old_masks;
    auto walk = [&](auto&& self, State s) -> void {
        if (s.terminal()) return;
        int amount = s.raise_size(false, rng), mask = legal_mask(s, amount);
        auto key = bucket(s, 0, mask);
        auto found = masks.emplace(key, mask);
        assert(found.second || found.first->second == mask);
        assert(key_actions(key) == mask);
        old_masks[bucket(s, 0)] |= 1 << mask;
        for (int action = 0; action < 3; ++action) if (mask & (1 << action)) {
            auto next = s; next.act(action, amount); self(self, next);
        }
    };
    walk(walk, State{});
    int collisions = 0;
    for (const auto& entry : old_masks) collisions += (entry.second & (entry.second - 1)) != 0;
    assert(collisions == 24 && masks.size() > old_masks.size());
    std::cout << "Full betting tree: " << old_masks.size() << " old contexts, " << masks.size()
              << " action-aware contexts; 24 old collisions, zero new collisions\n";
    std::cout << "C++ engine, rehash, checkpoint/resume and memory rollback checks passed\n";
}

void snapshot(std::ostream& out, const State& state, const Cards& cards) {
    out << "{\"street\":" << state.street << ",\"actor\":" << state.actor
        << ",\"stacks\":[" << state.stacks[0] << ',' << state.stacks[1]
        << "],\"bets\":[" << state.bets[0] << ',' << state.bets[1]
        << "],\"pot\":" << state.pot() << ",\"minimum\":";
    int high = std::max(state.bets[0], state.bets[1]);
    int effective = std::min(state.stacks[0] + state.bets[0], state.stacks[1] + state.bets[1]);
    int minimum = std::min(effective, high + state.min_raise);
    out << (state.can_raise(minimum) ? minimum : -1) << ",\"bucket\":";
    if (state.terminal()) out << "null";
    else {
        std::mt19937_64 rng(1);
        int mask = legal_mask(state, state.raise_size(false, rng));
        out << bucket_json(bucket(state, card_bucket(cards, state.actor, state.street, 20, rng), mask), false, true);
    }
    std::array<int, 7> a{}, b{};
    a[0] = cards[0]; a[1] = cards[1]; b[0] = cards[2]; b[1] = cards[3];
    for (int i = 0; i < 5; ++i) a[i + 2] = b[i + 2] = cards[i + 4];
    auto sa = evaluate(a.data(), 7), sb = evaluate(b.data(), 7);
    out << ",\"payoff\":" << state.payoff(sa > sb ? 0 : sa < sb ? 1 : 2) << '}';
}

int main(int argc, char** argv) {
    if (argc == 1) { unit_tests(); return 0; }
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        if (std::string(argv[1]) == "eval") {
            std::array<int, 7> cards{};
            std::string token; int n = 0;
            while (in >> token) cards[n++] = card(token);
            std::cout << evaluate(cards.data(), n) << '\n';
        } else {
            Cards cards{};
            std::string token;
            for (int& c : cards) { in >> token; c = card(token); }
            State state;
            std::cout << '['; snapshot(std::cout, state, cards);
            int action, amount;
            while (in >> action >> amount) {
                state.act(action, amount); std::cout << ','; snapshot(std::cout, state, cards);
            }
            std::cout << "]\n";
        }
    }
}
