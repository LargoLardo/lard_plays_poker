#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include "poker.hpp"
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using namespace poker;
#ifndef POKER_TESTING
static volatile std::sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }
#endif

struct Node {
    std::array<double, 3> regret{}, strategy{};
    uint64_t visits = 0;
    void add(const Node& other) {
        for (int i = 0; i < 3; ++i) { regret[i] += other.regret[i]; strategy[i] += other.strategy[i]; }
        visits += other.visits;
    }
};
struct Entry { uint32_t key = 0; Node node; }; // key+1; zero marks an empty slot.
struct BudgetExceeded : std::runtime_error { using std::runtime_error::runtime_error; };

// A flat, growing table avoids per-node allocation and long string keys.
// The budget includes both tables during a rehash and reserved traversal space.
class Nodes {
    std::vector<Entry> entries;
    size_t used = 0, max_slots;
    static size_t hash(uint32_t key) {
        uint32_t h = key;
        h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16;
        return h;
    }
    size_t slot(uint32_t key) const {
        size_t i = hash(key) & (entries.size() - 1);
        while (entries[i].key && entries[i].key != key + 1) i = (i + 1) & (entries.size() - 1);
        return i;
    }
public:
    explicit Nodes(size_t bytes) {
        max_slots = 1024;
        while (max_slots <= bytes / sizeof(Entry) / 3) max_slots *= 2;
        // Peak at rehash: old capacity + new capacity = 1.5 * new capacity.
        if (max_slots * sizeof(Entry) * 3 / 2 > bytes) max_slots /= 2;
        if (max_slots < 1024) throw std::runtime_error("Memory budget too small");
        entries.resize(1024);
    }
    size_t size() const { return used; }
    size_t capacity_bytes() const { return entries.size() * sizeof(Entry); }
    size_t limit() const { return max_slots * 7 / 10; }
    const Node* get(uint32_t key) const {
        const auto& entry = entries[slot(key)];
        return entry.key ? &entry.node : nullptr;
    }
    bool prepare(size_t additional) {
        size_t needed = used + additional, target = entries.size();
        while (needed > target * 7 / 10 && target < max_slots) target *= 2;
        if (needed > target * 7 / 10) return false;
        if (target != entries.size()) {
            std::vector<Entry> replacement(target); // Allocate before touching the trained table.
            auto old = std::move(entries);
            entries = std::move(replacement);
            for (const auto& entry : old) if (entry.key) entries[slot(entry.key - 1)] = entry;
        }
        return true;
    }
    void add(uint32_t key, const Node& node) {
        auto& entry = entries[slot(key)];
        if (!entry.key) { entry.key = key + 1; ++used; }
        entry.node.add(node);
    }
    template<class F> void each(F callback) const {
        for (const auto& entry : entries) if (entry.key) callback(entry.key - 1, entry.node);
    }
};

struct Traversal {
    bool preflop;
    int samples;
    std::mt19937_64 rng;
    const Nodes* base_nodes;
    size_t delta_limit;
    std::unordered_map<uint32_t, Node> delta;
    std::array<std::array<int, 4>, 2> hands{};
    Cards cards{};
    int winner = 2;
    Traversal(bool pf, int count, uint64_t seed, size_t reserve, const Nodes* base = nullptr)
        : preflop(pf), samples(count), rng(seed), base_nodes(base), delta_limit(reserve) {
        delta.reserve(std::min(size_t(128), delta_limit));
    }
    double traverse(const State& state, int traverser) {
        if (state.terminal() || (preflop && state.street > 0)) {
            double value = state.payoff(winner);
            return traverser == 0 ? value : -value;
        }
        int& hand = hands[state.actor][state.street];
        if (hand < 0) hand = card_bucket(cards, state.actor, state.street, samples, rng);
        int amount = state.raise_size(preflop, rng);
        int mask = legal_mask(state, amount);
        uint32_t key = bucket(state, hand, mask);
        auto it = delta.find(key);
        if (it == delta.end()) {
            if (delta.size() >= delta_limit) throw BudgetExceeded("Traversal memory reserve reached");
            it = delta.emplace(key, Node{}).first;
        }
        Node& change = it->second;
        const Node* base = base_nodes->get(key);
        std::array<bool, 3> legal{bool(mask & 1), true, bool(mask & 4)};
        std::array<double, 3> strategy{}, values{};
        double positive = 0;
        int count = 0;
        for (int action = 0; action < 3; ++action) if (legal[action]) {
            strategy[action] = std::max(0.0, change.regret[action] + (base ? base->regret[action] : 0));
            positive += strategy[action]; ++count;
        }
        for (int action = 0; action < 3; ++action) if (legal[action])
            strategy[action] = positive > 0 ? strategy[action] / positive : 1.0 / count;
        if (state.actor == traverser) {
            double expected = 0;
            for (int action = 0; action < 3; ++action) if (legal[action]) {
                State next = state;
                next.act(action, amount);
                values[action] = traverse(next, traverser);
                expected += strategy[action] * values[action];
            }
            for (int action = 0; action < 3; ++action) if (legal[action]) {
                change.regret[action] += values[action] - expected;
            }
            return expected;
        }
        // Average at sampled opponent nodes; regret updates stay at traverser nodes.
        ++change.visits;
        for (int action = 0; action < 3; ++action) if (legal[action])
            change.strategy[action] += strategy[action];
        double roll = std::generate_canonical<double, 53>(rng);
        int chosen = 1;
        for (int action = 0; action < 3; ++action) if (legal[action]) {
            chosen = action;
            roll -= strategy[action];
            if (roll <= 0) break;
        }
        State next = state;
        next.act(chosen, amount);
        return traverse(next, traverser);
    }
    void deal() {
        for (auto& row : hands) row.fill(-1);
        std::array<int, 52> deck{};
        for (int i = 0; i < 52; ++i) deck[i] = i;
        for (int i = 0; i < 9; ++i) {
            int j = std::uniform_int_distribution<int>(i, 51)(rng);
            std::swap(deck[i], deck[j]); cards[i] = deck[i];
        }
        std::array<int, 7> a{}, b{};
        a[0] = cards[0]; a[1] = cards[1]; b[0] = cards[2]; b[1] = cards[3];
        for (int i = 0; i < 5; ++i) a[i + 2] = b[i + 2] = cards[i + 4];
        auto sa = evaluate(a.data(), 7), sb = evaluate(b.data(), 7);
        winner = sa > sb ? 0 : sa < sb ? 1 : 2;
    }
};

struct Trainer : Traversal {
    int version = 4, workers = 1;
    uint64_t chunk_size = 64, iterations = 0;
    Nodes nodes;
    size_t max_nodes;
    Trainer(bool pf, int count, uint64_t seed, size_t budget, size_t limit)
        : Traversal(pf, count, seed, budget / 8 / 256), nodes(budget * 7 / 8),
          max_nodes(limit ? std::min(limit, nodes.limit()) : nodes.limit()) {
        base_nodes = &nodes;
    }
    void prepare_delta() {
        size_t additional = 0;
        for (const auto& entry : delta) additional += nodes.get(entry.first) == nullptr;
        if (nodes.size() + additional > max_nodes || !nodes.prepare(additional))
            throw BudgetExceeded("Node/memory limit reached");
    }
    void apply_delta(uint64_t count) {
        for (const auto& entry : delta) nodes.add(entry.first, entry.second);
        iterations += count;
    }
    bool step() {
        if (version < 3) throw std::runtime_error("Legacy checkpoint buckets merge legal actions; start a fresh run with a new --output");
        auto previous_rng = rng;
        delta.clear();
        try {
            deal();
            traverse(State{}, int(iterations % 2));
            prepare_delta();
        } catch (const BudgetExceeded& error) {
            // Discard the whole unfinished hand, including its random draws.
            rng = previous_rng; delta.clear();
            std::cerr << error.what() << "; stopping before the unfinished hand. Increase the limit to resume.\n";
            return false;
        } catch (const std::bad_alloc&) {
            rng = previous_rng; delta.clear();
            std::cerr << "Allocation refused; stopping before the unfinished hand.\n";
            return false;
        }
        apply_delta(1);
        return true;
    }
};

// Workers read one frozen table and keep bounded local updates. Only the
// coordinator merges, so there are no concurrent writes or model copies.
class ParallelTrainer {
    struct Job {
        Traversal traversal;
        uint64_t offset = 0, count = 0;
        std::exception_ptr error;
        Job(const Trainer& trainer, size_t reserve)
            : traversal(trainer.preflop, trainer.samples, 1, reserve, &trainer.nodes) {}
    };
    Trainer& trainer;
    std::vector<std::unique_ptr<Job>> jobs;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable start, finished;
    uint64_t generation = 0;
    std::atomic<size_t> next_job{0};
    size_t completed = 0;
    bool stopping = false;
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        start.notify_all();
        for (auto& thread : threads) thread.join();
    }
public:
    explicit ParallelTrainer(Trainer& source) : trainer(source) {
        size_t job_count = size_t(trainer.workers) * 4;
        size_t reserve = trainer.delta_limit / (2 * job_count);
        if (!reserve) throw std::runtime_error("Memory budget too small for worker count");
        for (size_t i = 0; i < job_count; ++i) jobs.push_back(std::make_unique<Job>(trainer, reserve));
        trainer.delta_limit /= 2; // Other half covers all worker deltas together.
        try {
            for (int i = 0; i < trainer.workers; ++i) threads.emplace_back([this] {
                uint64_t seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        start.wait(lock, [&] { return stopping || generation != seen; });
                        if (stopping) return;
                        seen = generation;
                    }
                    for (;;) {
                        size_t index = next_job.fetch_add(1, std::memory_order_relaxed);
                        if (index >= jobs.size()) break;
                        auto& job = *jobs[index];
                        try {
                            for (uint64_t hand = 0; hand < job.count; ++hand) {
                                job.traversal.deal();
                                job.traversal.traverse(State{}, int((job.offset + hand) % 2));
                            }
                        } catch (...) { job.error = std::current_exception(); }
                    }
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        ++completed;
                    }
                    finished.notify_one();
                }
            });
        } catch (...) { stop(); throw; }
    }
    ~ParallelTrainer() { stop(); }
    bool step(uint64_t count) {
        auto previous_rng = trainer.rng;
        trainer.delta.clear();
        auto reject = [&](const char* reason) {
            trainer.rng = previous_rng; trainer.delta.clear();
            for (auto& job : jobs) job->traversal.delta.clear();
            std::cerr << reason << "; stopping before the unfinished batch. Increase the limit to resume.\n";
            return false;
        };
        {
            std::unique_lock<std::mutex> lock(mutex);
            uint64_t offset = trainer.iterations, remaining = count;
            for (auto& job : jobs) {
                // Four logical tasks per core balance variable traversal cost and
                // performance/efficiency cores without scheduler-dependent RNG.
                job->offset = offset; job->count = std::min(remaining, (trainer.chunk_size + 3) / 4);
                job->error = nullptr; job->traversal.delta.clear();
                if (job->count) job->traversal.rng.seed(trainer.rng());
                offset += job->count; remaining -= job->count;
            }
            completed = 0; next_job.store(0, std::memory_order_relaxed); ++generation;
            start.notify_all();
            finished.wait(lock, [&] { return completed == threads.size(); });
        }
        try {
            for (const auto& job : jobs) {
                if (job->error) std::rethrow_exception(job->error);
                for (const auto& entry : job->traversal.delta) trainer.delta[entry.first].add(entry.second);
            }
            trainer.prepare_delta();
        } catch (const BudgetExceeded& error) { return reject(error.what()); }
          catch (const std::bad_alloc&) { return reject("Allocation refused"); }
        trainer.apply_delta(count);
        return true;
    }
};

// Explicit little-endian fields: no struct padding or native-endian checkpoint data.
void write_uint(std::ostream& out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out.put(char((value >> (i * 8)) & 255));
}
uint64_t read_uint(std::istream& in, int bytes) {
    uint64_t value = 0;
    for (int i = 0; i < bytes; ++i) {
        int byte = in.get();
        if (byte == EOF) throw std::runtime_error("Truncated checkpoint");
        value |= uint64_t(byte) << (i * 8);
    }
    return value;
}
void write_double(std::ostream& out, double value) {
    uint64_t bits; std::memcpy(&bits, &value, 8); write_uint(out, bits, 8);
}
double read_double(std::istream& in) {
    uint64_t bits = read_uint(in, 8); double value; std::memcpy(&value, &bits, 8);
    if (!std::isfinite(value)) throw std::runtime_error("Invalid checkpoint numeric value");
    return value;
}
void make_parent(const fs::path& path) { if (path.has_parent_path()) fs::create_directories(path.parent_path()); }
void finish_file(std::ofstream& out, const fs::path& temporary, const fs::path& path) {
    out.flush();
    if (!out) throw std::runtime_error("Failed writing " + temporary.string());
    out.close();
    if (!out) throw std::runtime_error("Failed closing " + temporary.string());
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::system_error(int(GetLastError()), std::system_category(), "Cannot replace checkpoint");
#else
    fs::rename(temporary, path);
#endif
}
void save(const Trainer& trainer, const fs::path& path) {
    make_parent(path);
    fs::path temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot write " + temporary.string());
    out.write("LARDCPP", 7); out.put(char('0' + trainer.version));
    write_uint(out, trainer.preflop, 4); write_uint(out, trainer.samples, 4);
    if (trainer.version >= 4) { write_uint(out, trainer.workers, 4); write_uint(out, trainer.chunk_size, 4); }
    write_uint(out, trainer.iterations, 8); write_uint(out, trainer.nodes.size(), 8);
    std::ostringstream state; state << trainer.rng;
    auto text = state.str(); write_uint(out, text.size(), 4); out << text;
    trainer.nodes.each([&](uint32_t key, const Node& node) {
        write_uint(out, key, 4);
        for (double v : node.regret) write_double(out, v);
        for (double v : node.strategy) write_double(out, v);
        write_uint(out, node.visits, 8);
    });
    finish_file(out, temporary, path);
}
void save_snapshot(const Trainer& trainer, const fs::path& output) {
    auto directory = output.parent_path() / (output.stem().string() + "-snapshots");
    auto extension = output.has_extension() ? output.extension().string() : ".bin";
    auto path = directory / ("iter-" + std::to_string(trainer.iterations) + extension);
    if (fs::exists(path)) {
        std::cout << "Keeping existing snapshot " << path << '\n';
        return;
    }
    save(trainer, path);
    std::cout << "Saved snapshot " << path << '\n';
}
void load(Trainer& trainer, const fs::path& path, bool mode_set, bool samples_set, bool reset_average = false,
          bool workers_set = false, bool chunk_set = false) {
    std::ifstream in(path, std::ios::binary);
    char magic[8]; in.read(magic, 8);
    if (!in || std::string(magic, 7) != "LARDCPP" || magic[7] < '1' || magic[7] > '4')
        throw std::runtime_error("Invalid C++ checkpoint");
    trainer.version = magic[7] - '0';
    auto mode = read_uint(in, 4), samples = read_uint(in, 4);
    if (mode > 1 || samples < 1 || samples > 1'000'000) throw std::runtime_error("Invalid checkpoint settings");
    if ((mode_set && bool(mode) != trainer.preflop) || (samples_set && int(samples) != trainer.samples))
        throw std::runtime_error("Resume mode/sample count differs from checkpoint");
    trainer.preflop = bool(mode); trainer.samples = int(samples);
    if (trainer.version >= 4) {
        auto workers = read_uint(in, 4), chunk = read_uint(in, 4);
        if (workers < 1 || workers > 256 || chunk < 1 || chunk > 1'000'000)
            throw std::runtime_error("Invalid checkpoint worker settings");
        if (!workers_set) trainer.workers = int(workers);
        if (!chunk_set) trainer.chunk_size = chunk;
    }
    trainer.iterations = read_uint(in, 8);
    auto count = read_uint(in, 8), length = read_uint(in, 4);
    if (length > 20'000 || count > trainer.max_nodes || !trainer.nodes.prepare(size_t(count)))
        throw std::runtime_error("Checkpoint exceeds memory/node limit or has invalid metadata");
    std::string text(size_t(length), '\0'); in.read(text.data(), std::streamsize(length));
    std::istringstream state(text);
    if (!in || !(state >> trainer.rng)) throw std::runtime_error("Invalid checkpoint RNG state");
    for (uint64_t i = 0; i < count; ++i) {
        auto key = uint32_t(read_uint(in, 4));
        int street = (key >> 17) & 3, hand = key & 65535;
        bool invalid = (trainer.version < 3 && key >= (1U << 30)) || ((key >> 19) & 7) > (street == 1 ? 3 : 4)
            || ((key >> 22) & 7) > (street == 0 ? 6 : 3) || ((key >> 27) & 3) > 2
            || hand >= (street == 0 ? 338 : street == 1 ? 768 : street == 2 ? 3072 : 64);
        if (invalid || trainer.nodes.get(key)) throw std::runtime_error("Invalid/duplicate checkpoint bucket");
        Node node;
        for (double& v : node.regret) v = read_double(in);
        for (double& v : node.strategy) { v = read_double(in); if (v < 0) throw std::runtime_error("Negative strategy weight"); }
        node.visits = read_uint(in, 8);
        if (reset_average) { node.strategy.fill(0); node.visits = 0; }
        trainer.nodes.add(key, node);
    }
    if (in.peek() != EOF) throw std::runtime_error("Unexpected trailing checkpoint data");
}
std::string quoted(const std::string& text) {
    std::string out = "\"";
    for (char c : text) { if (c == '"' || c == '\\') out += '\\'; out += c; }
    return out + '"';
}
void export_web(const Trainer& trainer, const fs::path& directory) {
    fs::create_directories(directory);
    for (int post = 0; post < 2; ++post) {
        auto path = directory / (post ? "postflop-model.json" : "preflop-model.json");
        fs::path temporary = path.string() + ".tmp";
        std::ofstream out(temporary); out << std::setprecision(17) << '{';
        bool first = true;
        trainer.nodes.each([&](uint32_t key, const Node& node) {
            if (bool((key >> 17) & 3) != bool(post)) return;
            double total = node.strategy[0] + node.strategy[1] + node.strategy[2];
            if (total <= 0) return;
            if (!first) out << ',';
            first = false;
            out << quoted(bucket_json(key, !post, trainer.version >= 3)) << ":[";
            for (int a = 0; a < 3; ++a) { if (a) out << ','; out << node.strategy[a] / total; }
            out << ',' << node.visits << ']';
        });
        out << '}'; finish_file(out, temporary, path);
    }
    std::cout << "Exported browser models to " << directory << '\n';
}

uint64_t number(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a nonnegative integer: " + value);
    size_t end; auto result = std::stoull(value, &end);
    if (end != value.size()) throw std::runtime_error("Invalid integer: " + value);
    return result;
}

#ifndef POKER_TESTING
int main(int argc, char** argv) {
    try {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "Requires IEEE 754 doubles");
        uint64_t iterations = 100'000, seed = 1, memory_mb = 256, max_nodes = 0, checkpoint_every = 60, snapshot_every = 0;
        int samples = 100, workers = 1;
        uint64_t chunk_size = 64;
        bool preflop = false, mode_set = false, samples_set = false, reset_average = false, workers_set = false, chunk_set = false;
        fs::path output, resume, export_dir;
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--help") {
                std::cout << "Standalone heads-up Hold'em external-sampling MCCFR\n"
                    "  --iterations N        Additional hands (default 100000)\n"
                    "  --mode full|preflop   Full game or preflop check-through\n"
                    "  --samples N           Postflop equity samples (default 100)\n"
                    "  --workers N           CPU workers (default 1; 0 uses all cores)\n"
                    "  --chunk-size N        Hands per worker between merges (default 64)\n"
                    "  --memory-mb N         Node/rehash/traversal budget (default 256)\n"
                    "  --max-nodes N         Optional lower node cap\n"
                    "  --seed N              Reproducible fresh run (default 1)\n"
                    "  --output PATH         Binary checkpoint (default nodesets/cpp/<mode>.bin)\n"
                    "  --resume PATH         Load checkpoint, including mode/samples/RNG\n"
                    "  --reset-average       Discard averages/visits on a compatible resume\n"
                    "  --checkpoint-every N  Seconds between atomic saves (default 60)\n"
                    "  --snapshot-every N    Retain a separate checkpoint every N total hands (0 disables)\n"
                    "  --export DIR          Write compatible browser JSON after training\n";
                return 0;
            }
            if (flag == "--reset-average") { reset_average = true; continue; }
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + flag);
            std::string value = argv[++i];
            if (flag == "--iterations") iterations = number(value);
            else if (flag == "--seed") seed = number(value);
            else if (flag == "--memory-mb") memory_mb = number(value);
            else if (flag == "--max-nodes") max_nodes = number(value);
            else if (flag == "--checkpoint-every") checkpoint_every = number(value);
            else if (flag == "--snapshot-every") snapshot_every = number(value);
            else if (flag == "--workers") {
                auto count = number(value);
                if (count > 256) throw std::runtime_error("Workers must be 0..256");
                workers = count ? int(count) : int(std::max(1U, std::min(256U, std::thread::hardware_concurrency())));
                workers_set = true;
            } else if (flag == "--chunk-size") {
                chunk_size = number(value);
                if (!chunk_size || chunk_size > 1'000'000) throw std::runtime_error("Chunk size must be 1..1000000");
                chunk_set = true;
            }
            else if (flag == "--samples") {
                auto count = number(value);
                if (!count || count > 1'000'000) throw std::runtime_error("Samples must be 1..1000000");
                samples = int(count); samples_set = true;
            } else if (flag == "--mode") {
                if (value != "full" && value != "preflop") throw std::runtime_error("Mode must be full or preflop");
                preflop = value == "preflop"; mode_set = true;
            } else if (flag == "--output") output = value;
            else if (flag == "--resume") resume = value;
            else if (flag == "--export") export_dir = value;
            else throw std::runtime_error("Unknown option " + flag);
        }
        if (memory_mb < 4 || memory_mb > 1'048'576) throw std::runtime_error("Memory budget must be 4..1048576 MiB");
        Trainer trainer(preflop, samples, seed, size_t(memory_mb) * 1024 * 1024, size_t(max_nodes));
        trainer.workers = workers; trainer.chunk_size = chunk_size;
        if (reset_average && resume.empty()) throw std::runtime_error("--reset-average requires --resume");
        if (!resume.empty()) load(trainer, resume, mode_set, samples_set, reset_average, workers_set, chunk_set);
        if (trainer.version < 3 && iterations)
            throw std::runtime_error("Legacy checkpoint buckets merge legal actions; start a fresh run with a new --output. Use --iterations 0 to inspect/export the old model.");
        if (iterations > std::numeric_limits<uint64_t>::max() - trainer.iterations)
            throw std::runtime_error("Iteration count would overflow the checkpoint counter");
        if (trainer.version >= 3) trainer.version = 4;
        bool output_set = !output.empty();
        if (output.empty()) output = resume.empty() ? fs::path(trainer.preflop ? "nodesets/cpp/preflop.bin" : "nodesets/cpp/full.bin") : resume;
        if (resume.empty() && fs::exists(output)) throw std::runtime_error("Output already exists; use --resume or a new --output");
        std::signal(SIGINT, on_signal); std::signal(SIGTERM, on_signal);
        auto start = Clock::now(), last_save = start, last_report = start;
        uint64_t before = trainer.iterations;
        std::cout << (trainer.preflop ? "Preflop" : "Full-game") << ": " << trainer.workers << " worker(s), " << memory_mb << " MiB budget, "
                  << trainer.max_nodes << " node limit; checkpoint " << output << '\n';
        std::unique_ptr<ParallelTrainer> parallel;
        if (trainer.workers > 1 && iterations) parallel = std::make_unique<ParallelTrainer>(trainer);
        while (trainer.iterations - before < iterations && !interrupted) {
            uint64_t count = parallel ? std::min(iterations - (trainer.iterations - before), uint64_t(trainer.workers) * trainer.chunk_size) : 1;
            if (snapshot_every) count = std::min(count, snapshot_every - trainer.iterations % snapshot_every);
            if (!(parallel ? parallel->step(count) : trainer.step())) break;
            if (snapshot_every && trainer.iterations % snapshot_every == 0)
                save_snapshot(trainer, output);
            auto now = Clock::now();
            if (std::chrono::duration<double>(now - last_save).count() >= checkpoint_every) {
                save(trainer, output); last_save = now;
            }
            if (std::chrono::duration<double>(now - last_report).count() >= 5) {
                std::cout << trainer.iterations << " hands, " << trainer.nodes.size() << " nodes\n" << std::flush;
                last_report = now;
            }
        }
        if (trainer.version >= 3 || output_set) save(trainer, output);
        if (!export_dir.empty()) export_web(trainer, export_dir);
        double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << (trainer.version < 3 && !output_set ? "Loaded " : "Saved ") << trainer.iterations << " completed hands, " << trainer.nodes.size() << " nodes; "
                  << std::fixed << std::setprecision(1) << (trainer.iterations - before) / std::max(seconds, .000001)
                  << " hands/s; table " << trainer.nodes.capacity_bytes() / (1024.0 * 1024) << " MiB\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'; return 1;
    }
}
#endif
