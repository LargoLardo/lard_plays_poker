// Reuse the existing deal/evaluator, portable checkpoint I/O and compiler setup.
#ifndef POKER_TESTING
#define POKER_TESTING
#endif
#include "trainer.cpp"
#include "model_v5.hpp"
#include <map>

namespace v5 {
template<class F> void parallel_for(size_t count, int workers, F work) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    std::exception_ptr error;
    std::mutex mutex;
    auto run = [&] {
        try {
            for (;;) {
                size_t start = next.fetch_add(64);
                if (start >= count) break;
                for (size_t i = start; i < std::min(count, start + 64); ++i) work(i);
            }
        } catch (...) { std::lock_guard<std::mutex> lock(mutex); if (!error) error = std::current_exception(); }
    };
    try { for (int i = 0; i < workers; ++i) threads.emplace_back(run); }
    catch (...) { for (auto& t : threads) t.join(); throw; }
    for (auto& t : threads) t.join();
    if (error) std::rethrow_exception(error);
}
inline void write_abstraction(std::ostream& out, const Abstraction& value) {
    write_uint(out, value.samples, 4); write_uint(out, value.seed, 8); write_uint(out, value.examples, 8);
    for (const auto& street : value.centers) {
        write_uint(out, street.size(), 4);
        for (const auto& center : street) for (float component : center) write_double(out, component);
    }
}
inline Abstraction read_abstraction(std::istream& in) {
    Abstraction value;
    auto samples = read_uint(in, 4);
    if (!samples || samples > 1'000'000) throw std::runtime_error("Invalid abstraction sample budget");
    value.samples = int(samples); value.seed = read_uint(in, 8); value.examples = read_uint(in, 8);
    for (auto& street : value.centers) {
        auto count = read_uint(in, 4);
        if (!count || count > 4096) throw std::runtime_error("Abstraction must have 1..4096 clusters per street");
        street.resize(size_t(count));
        for (auto& center : street) for (float& component : center) {
            double x = read_double(in);
            if (x < 0 || x > 1.00001) throw std::runtime_error("Invalid abstraction feature");
            component = float(x);
        }
    }
    return value;
}
inline Abstraction load_abstraction(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    char magic[8]; in.read(magic, 8);
    if (!in || std::string(magic, 8) != "LARDABS1") throw std::runtime_error("Invalid abstraction file");
    auto value = read_abstraction(in);
    if (in.peek() != EOF) throw std::runtime_error("Trailing abstraction data");
    return value;
}
inline Abstraction learn(int samples, uint64_t seed, size_t examples, const std::array<int, 3>& clusters, int workers, int rounds) {
    Abstraction result;
    result.samples = samples; result.seed = seed; result.examples = examples;
    for (int street = 1; street <= 3; ++street) {
        std::vector<Features> data(examples);
        parallel_for(examples, workers, [&](size_t index) {
            Random rng{mix(seed ^ (uint64_t(street) << 56) ^ index)};
            std::array<int, 52> deck{}; std::iota(deck.begin(), deck.end(), 0);
            Cards cards{};
            for (int i = 0; i < 9; ++i) { int j = i + rng.bounded(52 - i); std::swap(deck[i], deck[j]); cards[i] = deck[i]; }
            data[index] = features(canonical(cards, 0, street), samples, seed);
        });
        auto& centers = result.centers[size_t(street - 1)];
        centers.resize(size_t(clusters[size_t(street - 1)]));
        for (size_t k = 0; k < centers.size(); ++k) centers[k] = data[k * examples / centers.size()];
        std::vector<size_t> labels(examples);
        for (int round = 0; round < rounds; ++round) {
            parallel_for(examples, workers, [&](size_t i) { labels[i] = size_t(result.classify(data[i], street)); });
            std::vector<std::vector<size_t>> groups(centers.size());
            for (size_t i = 0; i < examples; ++i) groups[labels[i]].push_back(i);
            for (size_t k = 0; k < centers.size(); ++k) {
                if (groups[k].empty()) { centers[k] = data[size_t(mix(seed + k + uint64_t(round) * centers.size()) % examples)]; continue; }
                std::vector<float> values; values.reserve(groups[k].size());
                for (int dimension = 0; dimension < dimensions; ++dimension) {
                    if (dimension < 16) {
                        values.clear();
                        for (size_t i : groups[k]) values.push_back(data[i][dimension]);
                        auto middle = values.begin() + std::ptrdiff_t(values.size() / 2);
                        std::nth_element(values.begin(), middle, values.end()); centers[k][dimension] = *middle;
                    } else {
                        double sum = 0; for (size_t i : groups[k]) sum += data[i][dimension];
                        centers[k][dimension] = float(sum / groups[k].size());
                    }
                }
            }
            std::cerr << "Street " << street << ": clustering round " << round + 1 << '/' << rounds << '\n';
        }
    }
    return result;
}

struct Walk : ::Traversal {
    const Abstraction* abstraction;
    const Nodes* table;
    CardCache cache;
    std::unordered_map<Key, Node, Hash> updates;
    size_t reserve;
    uint64_t touches = 0;
    Walk(const Abstraction& assets, const Nodes* source, size_t capacity, size_t cache_entries)
        : ::Traversal(false, assets.samples, 1, 0), abstraction(&assets), table(source), reserve(capacity) { cache.limit = cache_entries; }
    double traverse(const Position& position, int traverser) {
        ++touches;
        const auto& state = position.state;
        if (state.terminal()) return state.payoff(winner) * (traverser == 0 ? 1 : -1);
        int& hand = hands[state.actor][state.street];
        if (hand < 0) hand = cache.hand(*abstraction, cards, state.actor, state.street);
        auto choices = actions(state);
        auto info = key(position, hand);
        auto found = updates.find(info);
        if (found == updates.end()) {
            if (updates.size() >= reserve) throw BudgetExceeded("V5 update buffer reached");
            found = updates.emplace(info, Node{}).first;
        }
        Node& change = found->second;
        const Node* base = table->get(info);
        std::array<double, action_count> strategy{}, values{};
        double total = 0;
        int legal = 0;
        for (int a = 0; a < action_count; ++a) if (choices.mask & (1 << a)) {
            strategy[a] = std::max(0.0, change.regret[a] + (base ? base->regret[a] : 0));
            total += strategy[a]; ++legal;
        }
        for (int a = 0; a < action_count; ++a) if (choices.mask & (1 << a)) strategy[a] = total > 0 ? strategy[a] / total : 1.0 / legal;
        if (state.actor == traverser) {
            double value = 0;
            for (int a = 0; a < action_count; ++a) if (choices.mask & (1 << a)) {
                auto next = position; next.act(a); values[a] = traverse(next, traverser); value += strategy[a] * values[a];
            }
            for (int a = 0; a < action_count; ++a) if (choices.mask & (1 << a)) change.regret[a] += values[a] - value;
            return value;
        }
        ++change.visits;
        for (int a = 0; a < action_count; ++a) if (choices.mask & (1 << a)) change.strategy[a] += strategy[a];
        double roll = std::generate_canonical<double, 53>(rng);
        int chosen = 1;
        for (int a = 0; a < action_count; ++a) if (choices.mask & (1 << a)) { chosen = a; roll -= strategy[a]; if (roll < 0) break; }
        auto next = position; next.act(chosen); return traverse(next, traverser);
    }
};
struct Model {
    Abstraction abstraction;
    Nodes nodes;
    uint64_t iterations = 0, touches = 0, discount_every = 10'000'000, periods = 0, chunk = 64;
    bool linear = true;
    int workers = 1;
    size_t maximum, reserve, cache_entries;
    std::mt19937_64 rng{1};
    Model(size_t budget, size_t cache_bytes, size_t limit = 0)
        : nodes(budget > cache_bytes + budget / 8 ? budget - cache_bytes - budget / 8 : 0),
          maximum(limit ? std::min(limit, nodes.limit()) : nodes.limit()), reserve(budget / 8 / 256), cache_entries(cache_bytes / 64) {}
    void merge(const std::unordered_map<Key, Node, Hash>& changes, uint64_t hands, uint64_t touched) {
        size_t added = 0;
        for (const auto& item : changes) added += nodes.get(item.first) == nullptr;
        if (nodes.size() + added > maximum || !nodes.prepare(added)) throw BudgetExceeded("V5 node/memory budget reached");
        if (touched > std::numeric_limits<uint64_t>::max() - touches) throw std::runtime_error("Traversal counter overflow");
        for (const auto& item : changes) nodes.add(item.first, item.second);
        iterations += hands; touches += touched;
        if (linear) {
            uint64_t next = touches / discount_every;
            if (next > periods) { nodes.discount(double(periods + 1) / double(next + 1)); periods = next; }
        }
    }
};
class Serial {
    Model& model;
    Walk walk;
public:
    explicit Serial(Model& source) : model(source), walk(model.abstraction, &model.nodes, model.reserve, model.cache_entries) {}
    bool step() {
        walk.updates.clear(); walk.touches = 0; walk.rng = model.rng;
        try {
            walk.deal(); walk.traverse(Position{}, int(model.iterations % 2));
            model.merge(walk.updates, 1, walk.touches);
        } catch (const BudgetExceeded& error) { std::cerr << error.what() << "; stopped before unfinished hand\n"; return false; }
          catch (const std::bad_alloc&) { std::cerr << "Allocation refused; stopped before unfinished hand\n"; return false; }
        model.rng = walk.rng;
        return true;
    }
};
class Pool {
    struct Job {
        Walk walk;
        uint64_t offset = 0, count = 0;
        std::exception_ptr error;
        Job(Model& model, size_t reserve, size_t cache) : walk(model.abstraction, &model.nodes, reserve, cache) {}
    };
    Model& model;
    std::vector<std::unique_ptr<Job>> jobs;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable start, done;
    std::atomic<size_t> next{0};
    uint64_t generation = 0;
    size_t completed = 0;
    bool stopping = false;
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        start.notify_all(); for (auto& t : threads) t.join();
    }
public:
    explicit Pool(Model& source) : model(source) {
        size_t count = size_t(model.workers) * 4;
        if (model.reserve / (count * 2) < 1) throw std::runtime_error("Update budget too small for workers");
        for (size_t i = 0; i < count; ++i) jobs.push_back(std::make_unique<Job>(model, model.reserve / (count * 2), model.cache_entries / count));
        try { for (int i = 0; i < model.workers; ++i) threads.emplace_back([this] {
            uint64_t seen = 0;
            for (;;) {
                { std::unique_lock<std::mutex> lock(mutex); start.wait(lock, [&] { return stopping || generation != seen; }); if (stopping) return; seen = generation; }
                for (;;) {
                    size_t index = next.fetch_add(1); if (index >= jobs.size()) break;
                    auto& job = *jobs[index];
                    try { for (uint64_t i = 0; i < job.count; ++i) { job.walk.deal(); job.walk.traverse(Position{}, int((job.offset + i) % 2)); } }
                    catch (...) { job.error = std::current_exception(); }
                }
                { std::lock_guard<std::mutex> lock(mutex); ++completed; } done.notify_one();
            }
        }); } catch (...) { stop(); throw; }
    }
    ~Pool() { stop(); }
    bool step(uint64_t count) {
        auto prior = model.rng;
        {
            std::unique_lock<std::mutex> lock(mutex);
            uint64_t remaining = count, offset = model.iterations;
            for (auto& job : jobs) {
                job->count = std::min(remaining, (model.chunk + 3) / 4); job->offset = offset;
                job->error = nullptr; job->walk.updates.clear(); job->walk.touches = 0;
                if (job->count) job->walk.rng.seed(model.rng());
                remaining -= job->count; offset += job->count;
            }
            next.store(0); completed = 0; ++generation; start.notify_all(); done.wait(lock, [&] { return completed == threads.size(); });
        }
        try {
            std::unordered_map<Key, Node, Hash> changes;
            uint64_t touched = 0;
            for (const auto& job : jobs) {
                if (job->error) std::rethrow_exception(job->error);
                touched += job->walk.touches;
                for (const auto& item : job->walk.updates) {
                    if (changes.size() >= model.reserve / 2 && changes.find(item.first) == changes.end()) throw BudgetExceeded("V5 merge buffer reached");
                    changes[item.first].add(item.second);
                }
            }
            model.merge(changes, count, touched);
        } catch (const BudgetExceeded& error) { model.rng = prior; std::cerr << error.what() << "; stopped before unfinished batch\n"; return false; }
          catch (const std::bad_alloc&) { model.rng = prior; std::cerr << "Allocation refused; stopped before unfinished batch\n"; return false; }
        return true;
    }
};
inline void save(const Model& model, const fs::path& path) {
    make_parent(path); fs::path temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write V5 checkpoint");
    out.write("LARDCPP5", 8); write_uint(out, 0, 4); write_uint(out, model.abstraction.samples, 4);
    write_uint(out, model.workers, 4); write_uint(out, model.chunk, 4);
    write_uint(out, model.iterations, 8); write_uint(out, model.nodes.size(), 8);
    write_uint(out, model.touches, 8); write_uint(out, model.linear, 4); write_uint(out, model.discount_every, 8); write_uint(out, model.periods, 8);
    std::ostringstream rng; rng << model.rng; auto text = rng.str(); write_uint(out, text.size(), 4); out << text;
    write_abstraction(out, model.abstraction);
    model.nodes.each([&](const Key& key, const Node& node) {
        write_uint(out, key.history, 8); write_uint(out, key.context, 4);
        for (double value : node.regret) write_double(out, value);
        for (double value : node.strategy) write_double(out, value);
        write_uint(out, node.visits, 8);
    });
    finish_file(out, temporary, path);
}
inline void load(Model& model, const fs::path& path) {
    std::ifstream in(path, std::ios::binary); char magic[8]; in.read(magic, 8);
    if (!in || std::string(magic, 8) != "LARDCPP5") throw std::runtime_error("V5 needs a fresh run or V5 checkpoint; old models remain available with --model v4");
    auto mode = read_uint(in, 4), samples = read_uint(in, 4), workers = read_uint(in, 4), chunk = read_uint(in, 4);
    if (mode || !workers || workers > 256 || !chunk || chunk > 1'000'000) throw std::runtime_error("Invalid V5 settings");
    model.workers = int(workers); model.chunk = chunk; model.iterations = read_uint(in, 8);
    auto count = read_uint(in, 8); model.touches = read_uint(in, 8);
    auto linear = read_uint(in, 4); model.discount_every = read_uint(in, 8); model.periods = read_uint(in, 8);
    if (linear > 1 || !model.discount_every || model.periods != (linear ? model.touches / model.discount_every : 0)) throw std::runtime_error("Invalid discount metadata");
    model.linear = bool(linear);
    auto length = read_uint(in, 4);
    if (length > 20'000 || count > model.maximum || !model.nodes.prepare(size_t(count))) throw std::runtime_error("V5 checkpoint exceeds memory/node budget");
    std::string text(size_t(length), '\0'); in.read(text.data(), std::streamsize(length)); std::istringstream rng(text);
    if (!in || !(rng >> model.rng)) throw std::runtime_error("Invalid V5 RNG state");
    model.abstraction = read_abstraction(in);
    if (samples != uint64_t(model.abstraction.samples)) throw std::runtime_error("Inconsistent V5 features");
    for (uint64_t i = 0; i < count; ++i) {
        Key key{read_uint(in, 8), uint32_t(read_uint(in, 4))};
        auto position = replay(key.history);
        int hand = int(key.context & 65535), street = position.state.street;
        if (position.state.terminal() || street > 3 || !(v5::key(position, hand) == key) || hand >= (street ? int(model.abstraction.centers[size_t(street - 1)].size()) : 338) || model.nodes.get(key))
            throw std::runtime_error("Invalid/duplicate V5 bucket");
        Node node;
        for (double& value : node.regret) value = read_double(in);
        for (double& value : node.strategy) { value = read_double(in); if (value < 0) throw std::runtime_error("Negative V5 strategy"); }
        int mask = actions(position.state).mask;
        for (int a = 0; a < action_count; ++a) if (!(mask & (1 << a)) && (node.regret[a] || node.strategy[a]))
            throw std::runtime_error("Weights on an illegal V5 action");
        node.visits = read_uint(in, 8); model.nodes.add(key, node);
    }
    if (in.peek() != EOF) throw std::runtime_error("Trailing V5 checkpoint data");
}
inline void export_web(const Model& model, const fs::path& directory) {
    struct Row { std::array<double, 3> weights{}; uint64_t visits = 0, sparse_visits = 0; };
    std::map<std::string, Row> rows;
    model.nodes.each([&](const Key& key, const Node& node) {
        if ((key.context >> 17) & 3) return;
        auto position = replay(key.history);
        int mask = int((key.context >> 19) & 31);
        auto compatible = ::bucket(position.state, int(key.context & 65535), 2 | (mask & 1) | (mask & 28 ? 4 : 0));
        auto& row = rows[bucket_json(compatible, true, true)];
        double total = std::accumulate(node.strategy.begin(), node.strategy.end(), 0.0);
        if (node.visits < 1000 || total <= 0) { row.sparse_visits = std::max(row.sparse_visits, node.visits); return; }
        row.visits += node.visits;
        row.weights[0] += node.strategy[0] / total * node.visits;
        row.weights[1] += node.strategy[1] / total * node.visits;
        row.weights[2] += (node.strategy[2] + node.strategy[3] + node.strategy[4]) / total * node.visits;
    });
    make_parent(directory / "preflop-model.json");
    auto path = directory / "preflop-model.json", temporary = fs::path(path.string() + ".tmp");
    std::ofstream out(temporary); out << std::setprecision(17) << '{'; bool first = true;
    for (const auto& item : rows) {
        if (!first) out << ',';
        first = false;
        out << quoted(item.first) << ":[";
        for (int a = 0; a < 3; ++a) { if (a) out << ','; out << (item.second.visits ? item.second.weights[a] / item.second.visits : 0); }
        out << ',' << (item.second.visits ? item.second.visits : std::min(uint64_t(999), item.second.sparse_visits)) << ']';
    }
    out << '}'; finish_file(out, temporary, path);
    path = directory / "postflop-model.json"; temporary = path.string() + ".tmp";
    std::ofstream post(temporary); post << "{}"; finish_file(post, temporary, path);
}
inline std::array<double, action_count> averaged(const Model& model, const Position& position, int hand, bool& trained, uint64_t& visits) {
    const auto* node = model.nodes.get(key(position, hand));
    auto choices = actions(position.state);
    std::array<double, action_count> weights{};
    double total = 0;
    for (int a = 0; a < action_count; ++a) if (choices.mask & (1 << a)) { weights[a] = node ? node->strategy[a] : 0; total += weights[a]; }
    trained = total > 0; visits = node ? node->visits : 0;
    if (!trained) for (int a = 0; a < action_count; ++a) if (choices.mask & (1 << a)) { weights[a] = 1; total += 1; }
    if (total) for (double& value : weights) value /= total;
    return weights;
}
// Compact stdin protocol keeps native inference persistent without a JSON dependency.
// hero1 hero2 board_count board... action_count [kind pot_fraction jam]... actor street
inline void infer(const Model& model) {
    CardCache cache; cache.limit = model.cache_entries;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            std::istringstream input(line); Cards cards{}; cards.fill(-1);
            std::string a, b; int n, count, actor, street;
            if (!(input >> a >> b >> n) || (n != 0 && n != 3 && n != 4 && n != 5)) throw std::runtime_error("Invalid visible cards");
            int first = card(a), second = card(b);
            std::array<bool, 52> seen{}; seen[first] = true; if (seen[second]) throw std::runtime_error("Duplicate card"); seen[second] = true;
            for (int i = 0; i < n; ++i) { std::string c; if (!(input >> c)) throw std::runtime_error("Missing board"); cards[4 + i] = card(c); if (seen[cards[4 + i]]) throw std::runtime_error("Duplicate card"); seen[cards[4 + i]] = true; }
            if (!(input >> count) || count < 0 || count > 40) throw std::runtime_error("Invalid history length");
            Position position; bool translated = false, available = true;
            for (int i = 0; i < count; ++i) {
                int kind, jam; double ratio;
                if (!(input >> kind >> ratio >> jam) || kind < 0 || kind > 2 || !std::isfinite(ratio) || ratio < 0 || ratio > 1000 || (jam != 0 && jam != 1)) throw std::runtime_error("Invalid observed action");
                if (!available) continue;
                auto choices = actions(position.state);
                int chosen = position.observe(kind, ratio, bool(jam));
                if (chosen < 0) { available = false; continue; }
                if (kind == 2) {
                    auto prior = replay(position.history >> 3);
                    int high = std::max(prior.state.bets[0], prior.state.bets[1]);
                    translated |= std::abs(double(choices.amounts[chosen] - high) / (prior.state.pot() + prior.state.to_call()) - ratio) > 1e-8;
                }
            }
            if (!(input >> actor >> street) || actor < 0 || actor > 1 || street < 0 || street > 3 || n != board_size(street)) throw std::runtime_error("Invalid decision position");
            std::string extra; if (input >> extra) throw std::runtime_error("Trailing inference data");
            cards[actor * 2] = first; cards[actor * 2 + 1] = second;
            if (!available || position.state.terminal() || position.state.actor != actor || position.state.street != street) {
                std::cout << "{\"unavailable\":true,\"weights\":[0,1,0,0,0],\"trained\":false}\n" << std::flush; continue;
            }
            int hand = cache.hand(model.abstraction, cards, actor, street);
            bool trained = false; uint64_t visits = 0; auto weights = averaged(model, position, hand, trained, visits);
            auto choices = actions(position.state);
            std::cout << std::setprecision(17) << "{\"weights\":[";
            for (int i = 0; i < action_count; ++i) { if (i) std::cout << ','; std::cout << weights[i]; }
            std::cout << "],\"amounts\":[";
            for (int i = 0; i < action_count; ++i) { if (i) std::cout << ','; std::cout << choices.amounts[i] / 2.0; }
            std::cout << "],\"mask\":" << choices.mask << ",\"bucket\":" << hand << ",\"history\":\"" << position.history
                << "\",\"visits\":" << visits << ",\"translated\":" << (translated ? "true" : "false") << ",\"trained\":" << (trained ? "true" : "false") << "}\n" << std::flush;
        } catch (const std::exception& error) { std::cout << "{\"error\":" << quoted(error.what()) << "}\n" << std::flush; }
    }
}
} // namespace v5

#ifndef MODEL_V5_TESTING
static volatile std::sig_atomic_t v5_interrupted = 0;
static void v5_signal(int) { v5_interrupted = 1; }
int main(int argc, char** argv) {
    try {
        fs::path abstraction, build_path, resume, output, exported;
        uint64_t hands = 100'000, seed = 1, budget = 4096, cache_mb = 256, max_nodes = 5'000'000, snapshot_every = 0, checkpoint_every = 60, discount_every = 10'000'000, chunk = 64;
        size_t examples = 50'000; std::array<int, 3> clusters{256, 512, 256};
        int samples = 512, rounds = 6, workers = int(std::max(1U, std::min(256U, std::thread::hardware_concurrency())));
        bool workers_set = false, chunk_set = false, samples_set = false, algorithm_set = false, discount_set = false, linear = true, inference = false;
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--help") {
                std::cout << "V5 heads-up 100bb Linear MCCFR\n"
                    "  --build-abstraction PATH  Learn frozen card clusters (50k examples/street)\n"
                    "  --abstraction PATH        Frozen asset required for a fresh training run\n"
                    "  --examples N --clusters F,T,R --cluster-rounds N --samples N --seed N\n"
                    "  --iterations N --resume PATH --output PATH --export DIR\n"
                    "  --workers N (0=all) --chunk-size N (64) --memory-mb N (4096)\n"
                    "  --cache-mb N (256) --max-nodes N (5000000; 0=budget limit)\n"
                    "  --algorithm linear|vanilla --discount-every N (10000000 nodes touched)\n"
                    "  --snapshot-every N --checkpoint-every SECONDS --infer\n";
                return 0;
            }
            if (flag == "--infer") { inference = true; continue; }
            if (++i >= argc) throw std::runtime_error("Missing value for " + flag);
            std::string value = argv[i];
            if (flag == "--build-abstraction") build_path = value;
            else if (flag == "--abstraction") abstraction = value;
            else if (flag == "--resume") resume = value;
            else if (flag == "--output") output = value;
            else if (flag == "--export") exported = value;
            else if (flag == "--iterations") hands = number(value);
            else if (flag == "--seed") seed = number(value);
            else if (flag == "--memory-mb") budget = number(value);
            else if (flag == "--cache-mb") cache_mb = number(value);
            else if (flag == "--max-nodes") max_nodes = number(value);
            else if (flag == "--snapshot-every") snapshot_every = number(value);
            else if (flag == "--checkpoint-every") checkpoint_every = number(value);
            else if (flag == "--discount-every") { discount_every = number(value); discount_set = true; }
            else if (flag == "--chunk-size") { chunk = number(value); chunk_set = true; }
            else if (flag == "--examples") examples = size_t(number(value));
            else if (flag == "--cluster-rounds") { auto n = number(value); if (!n || n > 100) throw std::runtime_error("Cluster rounds must be 1..100"); rounds = int(n); }
            else if (flag == "--samples") { auto n = number(value); if (!n || n > 1'000'000) throw std::runtime_error("Samples must be 1..1000000"); samples = int(n); samples_set = true; }
            else if (flag == "--workers") { auto n = number(value); if (n > 256) throw std::runtime_error("Workers must be 0..256"); workers = n ? int(n) : int(std::max(1U, std::min(256U, std::thread::hardware_concurrency()))); workers_set = true; }
            else if (flag == "--algorithm") { if (value != "linear" && value != "vanilla") throw std::runtime_error("Algorithm must be linear or vanilla"); linear = value == "linear"; algorithm_set = true; }
            else if (flag == "--clusters") {
                std::istringstream parts(value); std::string part;
                for (int j = 0; j < 3; ++j) { if (!std::getline(parts, part, ',')) throw std::runtime_error("Use --clusters flop,turn,river"); auto n = number(part); if (!n || n > 4096) throw std::runtime_error("Clusters must be 1..4096"); clusters[j] = int(n); }
                if (std::getline(parts, part, ',')) throw std::runtime_error("Use exactly three cluster counts");
            } else throw std::runtime_error("Unknown V5 option " + flag);
        }
        if (budget < 4 || budget > 1'048'576 || cache_mb >= budget * 7 / 8 || !chunk || chunk > 1'000'000 || !discount_every) throw std::runtime_error("Invalid memory, cache, chunk or discount budget");
        if (!build_path.empty()) {
            if (!resume.empty() || !output.empty() || !abstraction.empty() || inference || !exported.empty()) throw std::runtime_error("Build abstraction separately from training");
            if (fs::exists(build_path)) throw std::runtime_error("Abstraction already exists; choose a new path");
            if (examples < size_t(*std::max_element(clusters.begin(), clusters.end())) * 2 || examples > 1'000'000) throw std::runtime_error("Use 2x the largest cluster count to 1000000 examples per street");
            auto learned = v5::learn(samples, seed, examples, clusters, workers, rounds);
            make_parent(build_path); auto temporary = fs::path(build_path.string() + ".tmp"); std::ofstream out(temporary, std::ios::binary);
            out.write("LARDABS1", 8); v5::write_abstraction(out, learned); finish_file(out, temporary, build_path);
            std::cout << "Saved abstraction " << build_path << " (" << clusters[0] << '/' << clusters[1] << '/' << clusters[2] << " clusters)\n"; return 0;
        }
        v5::Model model(size_t(budget) * 1048576, size_t(cache_mb) * 1048576, size_t(max_nodes));
        if (!resume.empty()) {
            v5::load(model, resume);
            if (!abstraction.empty()) throw std::runtime_error("A resumed V5 checkpoint already contains its abstraction");
            if ((samples_set && samples != model.abstraction.samples) || (algorithm_set && linear != model.linear) || (discount_set && discount_every != model.discount_every)) throw std::runtime_error("Resume feature/algorithm settings differ; start a fresh output for an experiment");
        } else {
            if (abstraction.empty()) throw std::runtime_error("Build and supply --abstraction for a fresh V5 run");
            model.abstraction = v5::load_abstraction(abstraction);
            if (samples_set && samples != model.abstraction.samples) throw std::runtime_error("Samples must match the frozen abstraction");
            model.rng.seed(seed); model.linear = linear; model.discount_every = discount_every;
        }
        if (workers_set || resume.empty()) model.workers = workers;
        if (chunk_set || resume.empty()) model.chunk = chunk;
        if (inference) { if (resume.empty()) throw std::runtime_error("Inference requires --resume"); v5::infer(model); return 0; }
        if (hands > std::numeric_limits<uint64_t>::max() - model.iterations) throw std::runtime_error("Iteration counter overflow");
        if (output.empty()) output = resume.empty() ? fs::path("nodesets/cpp/full-v5.bin") : resume;
        if ((resume.empty() && fs::exists(output)) || (!resume.empty() && fs::exists(output) && fs::absolute(output) != fs::absolute(resume))) throw std::runtime_error("Output exists; use a fresh path or resume it explicitly");
        if (!resume.empty() && fs::absolute(output) != fs::absolute(resume) && output.extension() != ".bin") throw std::runtime_error("Use a .bin checkpoint output");
        std::signal(SIGINT, v5_signal); std::signal(SIGTERM, v5_signal);
        auto start = Clock::now(), last_save = start;
        uint64_t before = model.iterations;
        std::cerr << "V5 " << (model.linear ? "Linear" : "vanilla") << " MCCFR: " << model.workers << " workers, " << budget << " MiB budget, " << cache_mb << " MiB assignment cache, " << model.maximum << " node limit\n";
        if (hands) {
            std::unique_ptr<v5::Pool> pool;
            std::unique_ptr<v5::Serial> serial;
            if (model.workers == 1) serial = std::make_unique<v5::Serial>(model);
            else pool = std::make_unique<v5::Pool>(model);
            while (model.iterations - before < hands && !v5_interrupted) {
                uint64_t count = serial ? 1 : std::min(hands - (model.iterations - before), uint64_t(model.workers) * model.chunk);
                if (snapshot_every) count = std::min(count, snapshot_every - model.iterations % snapshot_every);
                if (!(serial ? serial->step() : pool->step(count))) break;
                if (snapshot_every && model.iterations % snapshot_every == 0) {
                    auto path = output.parent_path() / (output.stem().string() + "-snapshots") / ("iter-" + std::to_string(model.iterations) + ".bin");
                    if (!fs::exists(path)) v5::save(model, path);
                }
                auto now = Clock::now();
                if (checkpoint_every && now - last_save >= std::chrono::seconds(checkpoint_every)) { v5::save(model, output); last_save = now; std::cerr << model.iterations << " hands, " << model.nodes.size() << " nodes\n"; }
            }
        }
        // Export-only inspection never rewrites the source checkpoint.
        if (hands || resume.empty() || fs::absolute(output) != fs::absolute(resume)) v5::save(model, output);
        if (!exported.empty()) v5::export_web(model, exported);
        double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << model.iterations << " completed hands, " << model.nodes.size() << " nodes; " << (model.iterations - before) / std::max(seconds, .000001)
            << " hands/s; table " << model.nodes.bytes() / 1048576.0 << " MiB; " << model.touches << " nodes touched; " << model.periods << " discount periods\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
#endif
