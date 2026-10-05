#define MODEL_V5_TESTING
#include "train_v5.cpp"

static volatile std::sig_atomic_t arena_interrupted = 0;
static void arena_signal(int) { arena_interrupted = 1; }

struct Coverage {
    uint64_t decisions = 0, trained = 0, missing = 0, no_average = 0, preflop_runout = 0;
    uint64_t translated_actions = 0, off_tree_decisions = 0;
};

struct ArenaAgent {
    Trainer policy;
    std::shared_ptr<v5::Model> modern;
    v5::CardCache cache;
    v5::Position observed;
    bool observation_available = true;
    std::string baseline;
    bool legacy = false, swap_positions = false;
    Coverage coverage;
    uint64_t wins = 0, losses = 0, ties = 0;
    double net = 0;
    ArenaAgent(const fs::path& path, size_t budget, bool swap, std::shared_ptr<v5::Model> shared = {})
        : policy(false, 100, 1, budget, 0), swap_positions(swap) {
        if (path.string().rfind("baseline:", 0) == 0) {
            baseline = path.string().substr(9);
            if (baseline != "call" && baseline != "random" && baseline != "pot") throw std::runtime_error("Unknown baseline");
            if (swap_positions) throw std::runtime_error("Baselines do not have legacy seats");
            return;
        }
        std::ifstream in(path, std::ios::binary);
        char magic[8]; in.read(magic, 8);
        if (!in) throw std::runtime_error("Cannot read checkpoint " + path.string());
        if (std::string(magic, 8) == "LARDCPP5") {
            in.close();
            modern = std::move(shared);
            if (!modern) {
                modern = std::make_shared<v5::Model>(budget, std::min(size_t(64) * 1048576, budget / 8));
                v5::load(*modern, path);
            }
            cache.limit = modern->cache_entries;
        } else if (std::string(magic, 8) == "LARDPOL1") {
            auto schema = read_uint(in, 4), mode = read_uint(in, 4), samples = read_uint(in, 4);
            auto iterations = read_uint(in, 8), count = read_uint(in, 8);
            if ((schema != 1 && schema != 2) || mode > 1 || samples < 1 || samples > 1'000'000)
                throw std::runtime_error("Invalid Python policy metadata");
            policy.preflop = bool(mode); policy.samples = int(samples); policy.iterations = iterations;
            legacy = schema == 1;
            if (count > policy.max_nodes || !policy.nodes.prepare(size_t(count)))
                throw std::runtime_error("Policy exceeds arena memory budget");
            for (uint64_t i = 0; i < count; ++i) {
                auto key = uint32_t(read_uint(in, 4));
                if (policy.nodes.get(key)) throw std::runtime_error("Duplicate policy bucket");
                Node node;
                for (double& weight : node.strategy) {
                    weight = read_double(in);
                    if (weight < 0) throw std::runtime_error("Negative strategy weight");
                }
                node.visits = read_uint(in, 8);
                policy.nodes.add(key, node);
            }
            if (in.peek() != EOF) throw std::runtime_error("Trailing Python policy data");
        } else {
            in.close();
            load(policy, path, false, false);
            legacy = policy.version < 3;
        }
        if (swap_positions && !legacy)
            throw std::runtime_error("Seat correction applies only to legacy checkpoints");
    }

    int choose(const State& state, const Cards& cards, int hand, int& amount, std::mt19937_64& rng, bool call_fallback) {
        ++coverage.decisions;
        if (!baseline.empty()) {
            ++coverage.missing;
            if (baseline == "call") return 1;
            int high = std::max(state.bets[0], state.bets[1]);
            int maximum = state.stacks[state.actor] + state.bets[state.actor];
            int effective = std::min(maximum, state.stacks[1 - state.actor] + state.bets[1 - state.actor]);
            int minimum = std::min(effective, high + state.min_raise);
            amount = baseline == "random" && maximum > high ? std::uniform_int_distribution<int>(minimum, maximum)(rng)
                : std::min(maximum, high + state.pot() + state.to_call());
            auto mask = legal_mask(state, amount);
            std::vector<int> legal;
            for (int a = 0; a < 3; ++a) if (mask & (1 << a)) legal.push_back(a);
            return legal[std::uniform_int_distribution<size_t>(0, legal.size() - 1)(rng)];
        }
        if (modern) {
            auto choices = v5::actions(state);
            const v5::Node* node = nullptr;
            if (observation_available && !observed.state.terminal() && observed.state.actor == state.actor && observed.state.street == state.street) {
                int cluster = cache.hand(modern->abstraction, cards, state.actor, state.street);
                node = modern->nodes.get(v5::key(observed, cluster));
            } else ++coverage.off_tree_decisions;
            std::array<double, 5> weights{};
            double total = 0;
            for (int a = 0; a < 5; ++a) if (choices.mask & (1 << a)) { weights[a] = node ? node->strategy[a] : 0; total += weights[a]; }
            if (total > 0) ++coverage.trained;
            else {
                if (node) ++coverage.no_average; else ++coverage.missing;
                if (call_fallback) return 1;
                for (int a = 0; a < 5; ++a) if (choices.mask & (1 << a)) { weights[a] = 1; total += 1; }
            }
            double roll = std::generate_canonical<double, 53>(rng) * total;
            int chosen = 1;
            for (int a = 0; a < 5; ++a) if (choices.mask & (1 << a)) { chosen = a; roll -= weights[a]; if (roll < 0) break; }
            amount = choices.amounts[chosen];
            return chosen >= 2 ? 2 : chosen;
        }
        if (policy.preflop && state.street > 0) {
            ++coverage.preflop_runout;
            return 1; // Preflop-only policies were trained with check-through runouts.
        }
        int mask = legal_mask(state, amount);
        uint32_t key = bucket(state, hand, mask);
        if (legacy) key &= 0x3fffffffU;
        if (swap_positions) key ^= 1U << 16;
        const Node* node = policy.nodes.get(key);
        std::array<double, 3> weights{};
        double total = 0;
        if (node) for (int action = 0; action < 3; ++action) if (mask & (1 << action)) {
            weights[action] = node->strategy[action]; total += weights[action];
        }
        if (total > 0) ++coverage.trained;
        else {
            if (node) ++coverage.no_average;
            else ++coverage.missing;
            if (call_fallback) return 1;
            for (int action = 0; action < 3; ++action) if (mask & (1 << action)) {
                weights[action] = 1; total += 1;
            }
        }
        double roll = std::generate_canonical<double, 53>(rng) * total;
        int chosen = 1;
        for (int action = 0; action < 3; ++action) if (mask & (1 << action)) {
            chosen = action;
            roll -= weights[action];
            if (roll < 0) break;
        }
        return chosen;
    }
    void observe(int kind, double ratio, bool jam) {
        if (!modern || !observation_available) return;
        auto prior = observed;
        int chosen = observed.observe(kind, ratio, jam);
        if (chosen < 0) { observation_available = false; return; }
        if (kind == 2 && !jam) {
            auto choices = v5::actions(prior.state);
            int high = std::max(prior.state.bets[0], prior.state.bets[1]);
            double actual = double(choices.amounts[chosen] - high) / (prior.state.pot() + prior.state.to_call());
            if (std::abs(actual - ratio) > 1e-8) ++coverage.translated_actions;
        }
    }
    int samples() const { return modern ? modern->abstraction.samples : policy.samples; }
    void result(double value) {
        net += value;
        if (value > 0) ++wins;
        else if (value < 0) ++losses;
        else ++ties;
    }
};

double arena_hand(ArenaAgent& a, ArenaAgent& b, int a_seat, Traversal& deal,
                  uint64_t pair_seed, int samples, bool call_fallback) {
    State state;
    a.observed = {}; b.observed = {}; a.observation_available = b.observation_available = true;
    std::array<std::mt19937_64, 2> actions{std::mt19937_64(pair_seed), std::mt19937_64(pair_seed ^ 0x9e3779b97f4a7c15ULL)};
    while (!state.terminal()) {
        int seat = state.actor;
        ArenaAgent& agent = seat == a_seat ? a : b;
        int& hand = deal.hands[seat][state.street];
        if (hand < 0 && !agent.modern && agent.baseline.empty() && !(agent.policy.preflop && state.street > 0)) {
            // Same visible-card estimate for each seat/street in both duplicate legs.
            std::mt19937_64 features(pair_seed ^ (uint64_t(seat * 4 + state.street + 1) * 0xd1b54a32d192ed03ULL));
            hand = card_bucket(deal.cards, seat, state.street, samples, features);
        }
        int amount = state.raise_size(agent.policy.preflop, actions[seat]);
        int chosen = agent.choose(state, deal.cards, hand, amount, actions[seat], call_fallback);
        int high = std::max(state.bets[0], state.bets[1]);
        double ratio = chosen == 2 ? double(amount - high) / (state.pot() + state.to_call()) : 0;
        bool jam = chosen == 2 && amount == state.stacks[seat] + state.bets[seat];
        a.observe(chosen, ratio, jam); b.observe(chosen, ratio, jam);
        state.act(chosen, amount);
    }
    double value = state.payoff(deal.winner) * (a_seat == 0 ? 1 : -1);
    a.result(value); b.result(-value);
    return value;
}

void agent_json(std::ostream& out, const ArenaAgent& agent, uint64_t hands, double error, bool has_error) {
    double rate = hands ? agent.net * 100 / double(hands) : 0;
    const auto& c = agent.coverage;
    out << "{\"iterations\":";
    if (!agent.baseline.empty()) out << "null"; else out << (agent.modern ? agent.modern->iterations : agent.policy.iterations);
    out << ",\"model_version\":" << (agent.modern ? 5 : agent.policy.version) << ",\"feature_samples\":" << agent.samples()
        << ",\"baseline\":" << quoted(agent.baseline) << ",\"legacy\":" << (agent.legacy ? "true" : "false")
        << ",\"seat_correction\":" << (agent.swap_positions ? "true" : "false")
        << ",\"preflop_only\":" << (agent.policy.preflop ? "true" : "false")
        << ",\"net_bb\":" << agent.net << ",\"bb_per_100\":" << rate << ",\"ci95_bb_per_100\":";
    if (has_error) out << '[' << rate - error << ',' << rate + error << ']';
    else out << "null";
    out << ",\"wins\":" << agent.wins << ",\"losses\":" << agent.losses << ",\"ties\":" << agent.ties
        << ",\"coverage\":{\"decisions\":" << c.decisions << ",\"trained\":" << c.trained
        << ",\"missing\":" << c.missing << ",\"no_average\":" << c.no_average
        << ",\"preflop_runout\":" << c.preflop_runout << ",\"translated_actions\":" << c.translated_actions
        << ",\"off_tree_decisions\":" << c.off_tree_decisions << "}}";
}

int main(int argc, char** argv) {
    try {
        fs::path a_path, b_path;
        uint64_t requested = 10'000, seed = 1, memory_mb = 256;
        int samples = 0;
        bool swap_a = false, swap_b = false, call_fallback = false;
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            auto argument = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("Missing value for " + flag);
                return argv[i];
            };
            if (flag == "--a") a_path = argument();
            else if (flag == "--b") b_path = argument();
            else if (flag == "--hands") requested = number(argument());
            else if (flag == "--seed") seed = number(argument());
            else if (flag == "--samples") {
                auto count = number(argument());
                if (count > 1'000'000) throw std::runtime_error("Samples must be 0..1000000");
                samples = int(count);
            }
            else if (flag == "--memory-mb") memory_mb = number(argument());
            else if (flag == "--swap-a-legacy-positions") swap_a = true;
            else if (flag == "--swap-b-legacy-positions") swap_b = true;
            else if (flag == "--fallback") {
                auto mode = argument();
                if (mode != "uniform" && mode != "call") throw std::runtime_error("Fallback must be uniform or call");
                call_fallback = mode == "call";
            }
            else throw std::runtime_error("Unknown option " + flag);
        }
        if (a_path.empty() || b_path.empty() || requested < 2 || requested % 2)
            throw std::runtime_error("Choose two checkpoints and an even hand count of at least 2");
        if (memory_mb < 4 || memory_mb > 1'048'576) throw std::runtime_error("Invalid memory budget");
        ArenaAgent a(a_path, size_t(memory_mb) * 1024 * 1024, swap_a);
        // Mirrored matches share the large read-only policy, with separate actions/statistics.
        ArenaAgent b(b_path, size_t(memory_mb) * 1024 * 1024, swap_b, a_path == b_path ? a.modern : nullptr);
        if (!samples) samples = std::max(a.samples(), b.samples());
        Traversal deal(false, samples, seed, 0);
        uint64_t pairs = 0;
        double mean = 0, m2 = 0;
        auto start = Clock::now();
        std::signal(SIGINT, arena_signal); std::signal(SIGTERM, arena_signal);
        while (pairs < requested / 2 && !arena_interrupted) {
            deal.deal();
            uint64_t pair_seed = deal.rng();
            double first = arena_hand(a, b, 0, deal, pair_seed, samples, call_fallback);
            double second = arena_hand(a, b, 1, deal, pair_seed, samples, call_fallback);
            double value = (first + second) / 2;
            ++pairs;
            double difference = value - mean;
            mean += difference / double(pairs);
            m2 += difference * (value - mean);
        }
        uint64_t hands = pairs * 2;
        double error = pairs > 1 ? 1.96 * std::sqrt(std::max(0.0, m2) / double(pairs - 1) / double(pairs)) * 100 : 0;
        double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << std::setprecision(12) << "{\"hands\":" << hands << ",\"requested_hands\":" << requested
            << ",\"duplicate_pairs\":" << pairs << ",\"seed\":" << seed << ",\"samples\":" << samples
            << ",\"fallback\":\"" << (call_fallback ? "call" : "uniform") << "\",\"a\":";
        agent_json(std::cout, a, hands, error, pairs > 1);
        std::cout << ",\"b\":"; agent_json(std::cout, b, hands, error, pairs > 1);
        std::cout << ",\"elapsed_seconds\":" << seconds << ",\"hands_per_second\":" << hands / std::max(seconds, .000001)
            << ",\"interrupted\":" << (arena_interrupted ? "true" : "false") << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
