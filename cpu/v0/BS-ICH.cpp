
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <limits>
#include <cstdint>

using namespace std;

// Instance data structures
struct arc {
    int opd, opf, type;
    int poid;
};

struct arcalter {
    arc a, b;
    int num;
};

struct operation {
    int num_op;
    int num_mc;
    int dure;
    int td, tf;
    int tdp, tfp;
    int num_j;
    int type;
    double ds;
    int ops;
};

// Global variables for problem state
char instance_file[512] = "./la33.txt";
int kbest = 5000;          // Beam width K
int b_cmax = 500000;       // Best makespan initialized high

int nbjob = 0, nbmachine = 0, nboperation = 0;
vector<operation> list_opr;
vector<int> opr_mach[31];
vector<arc> arc_fix;
vector<arcalter> list_paire;
int id_paire = 0;

vector<int> dispo(31);
vector<int> tr(31);
vector<int> seq;
vector<int> ob[31];
vector<int> c;
vector<int> pere(302);
vector<int> td(302);
vector<int> tf(302);

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

class JobShopState {
public:
    std::vector<int> dispo;
    std::vector<int> tr;
    std::vector<int> seq;
    std::vector<int> c;
    std::vector<int> pere;
    std::vector<int> td;
    std::vector<int> tf;
    int cmax;
    double score;
    int64_t s_hash;

    JobShopState() : cmax(500000), score(0.0), s_hash(0) {}

    void save_state(
        const std::vector<int> &current_dispo, const std::vector<int> &current_tr,
        const std::vector<int> &current_seq, std::vector<int> current_ob[31],
        const std::vector<int> &current_c, const std::vector<int> &current_pere,
        const std::vector<int> &current_td, const std::vector<int> &current_tf, int64_t hash_code) 
    {
        s_hash = hash_code;
        dispo = current_dispo;
        tr = current_tr;
        seq = current_seq;
        c = current_c;
        pere = current_pere;
        td = current_td;
        tf = current_tf;
    }

    void load_state(
        std::vector<int> &current_dispo, std::vector<int> &current_tr,
        std::vector<int> &current_seq, std::vector<int> current_ob[31],
        std::vector<int> &current_c, std::vector<int> &current_pere,
        std::vector<int> &current_td, std::vector<int> &current_tf) const 
    {
        current_dispo = dispo;
        current_tr = tr;
        current_seq = seq;
        current_c = c;
        current_pere = pere;
        current_td = td;
        current_tf = tf;
    }
};

struct CompareTask {
    bool operator()(const JobShopState &t1, const JobShopState &t2) const {
        return t1.score < t2.score; // Max-Heap: worst candidate score at top
    }
};

int find_val(vector<int> &a, int v) {
    int ss = -1;
    if (!a.empty()) {
        auto res = std::find(a.begin(), a.end(), v);
        if (res != a.end()) {
            ss = std::distance(a.begin(), res);
        }
    }
    return ss;
}

// Function to resolve input file paths across benchmarks
bool resolve_instance_file(const char* input_arg, char* out_path) {
    // 1. Direct path check
    strcpy(out_path, input_arg);
    ifstream f1(out_path);
    if (f1.good()) return true;

    // 2. Format: ./laXX.txt
    snprintf(out_path, 512, "./la%s.txt", input_arg);
    ifstream f2(out_path);
    if (f2.good()) return true;

    // 3. Format: ./taXX.txt
    snprintf(out_path, 512, "./ta%s.txt", input_arg);
    ifstream f3(out_path);
    if (f3.good()) return true;

    // 4. Lawrence benchmark subfolder
    snprintf(out_path, 512, "../../lawrence_instances/la%s.txt", input_arg);
    ifstream f4(out_path);
    if (f4.good()) return true;

    // 5. Taillard benchmark subfolder
    snprintf(out_path, 512, "../../taillard_instances/ta%s", input_arg);
    ifstream f5(out_path);
    if (f5.good()) return true;

    snprintf(out_path, 512, "../../taillard_instances/ta%s", input_arg);
    ifstream f6(out_path);
    if (f6.good()) return true;

    return false;
}

void lecture_donnees(const char* path) {
    FILE *fichier = fopen(path, "r");
    if (!fichier) {
        fprintf(stderr, "Error opening instance file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    if (fscanf(fichier, "%d %d\n", &nbjob, &nbmachine) != 2) {
        fprintf(stderr, "Error reading header from %s\n", path);
        fclose(fichier);
        exit(EXIT_FAILURE);
    }

    nboperation = (nbjob * nbmachine) + 2;
    list_opr.clear();
    for (int i = 0; i <= 30; ++i) opr_mach[i].clear();
    arc_fix.clear();
    list_paire.clear();
    id_paire = 0;

    operation dummy;
    dummy.dure = 0; dummy.num_op = 0; dummy.num_mc = 0; dummy.num_j = 0;
    list_opr.push_back(dummy);

    int idf_op = 1;
    for (int i = 1; i <= nbjob; i++) {
        for (int j = 1; j <= nbmachine; j++) {
            int a = 0, b = 0;
            if (j == nbmachine) {
                fscanf(fichier, "%d %d\n", &a, &b);
            } else {
                fscanf(fichier, "%d %d", &a, &b);
            }
            operation p;
            p.dure = b;
            p.num_op = idf_op;
            p.num_mc = a + 1;
            p.num_j = i;
            list_opr.push_back(p);
            opr_mach[a + 1].push_back(idf_op);
            idf_op++;
        }
    }
    fclose(fichier);

    operation end_op;
    end_op.num_op = nbjob * nbmachine + 1;
    end_op.dure = 0; end_op.num_mc = 0; end_op.num_j = 0;
    list_opr.push_back(end_op);

    int lim = nbjob * nbmachine;
    for (int i1 = 1; i1 < lim; i1++) {
        if ((i1 % nbmachine) != 0) {
            arc o; o.opd = i1; o.opf = i1 + 1; o.poid = list_opr[i1].dure; o.type = 0;
            arc_fix.push_back(o);
        }
    }

    for (int i = 0; i < nbjob; i++) {
        arc a; a.opd = 0; a.opf = i * nbmachine + 1; a.poid = 0; a.type = 0;
        arc_fix.push_back(a);
    }

    for (int i = 1; i <= nbjob; i++) {
        arc a; a.opf = nbjob * nbmachine + 1; a.opd = i * nbmachine; a.poid = 0; a.type = 0;
        arc_fix.push_back(a);
    }
}

int solution_initial(int iteration) {
    // Reset state for construction
    dispo.assign(31, 0);
    tr.assign(31, 0);
    seq.clear();
    for (int i = 0; i <= 30; i++) ob[i].clear();
    c.clear();
    pere.assign(302, 0);
    td.assign(302, 0);
    tf.assign(302, 0);

    for (int i = 0; i < nbjob; i++) {
        c.push_back(i * nbmachine + 1);
    }

    std::priority_queue<JobShopState, std::vector<JobShopState>, CompareTask> beam;

    JobShopState initial_state;
    initial_state.save_state(dispo, tr, seq, ob, c, pere, td, tf, 0);
    initial_state.cmax = 0;
    initial_state.score = 0.0;
    beam.push(initial_state);

    int best_makespan = 500000;

    for (int step = 0; step < nbjob * nbmachine; step++) {
        std::priority_queue<JobShopState, std::vector<JobShopState>, CompareTask> next_beam;

        while (!beam.empty()) {
            JobShopState current = beam.top();
            beam.pop();

            for (size_t idx = 0; idx < current.c.size(); idx++) {
                JobShopState candidate = current;
                int op_candidate = current.c[idx];
                int mc = list_opr[op_candidate].num_mc;

                // Simulate scheduling op_candidate
                candidate.seq.push_back(op_candidate);
                candidate.c.erase(candidate.c.begin() + idx);

                if (op_candidate % nbmachine != 0) {
                    candidate.c.push_back(op_candidate + 1);
                }

                int prev_tr = candidate.tr[mc];
                candidate.tr[mc] = max(prev_tr, (op_candidate % nbmachine == 1) ? 0 : candidate.tf[op_candidate - 1]) + list_opr[op_candidate].dure;
                candidate.tf[op_candidate] = candidate.tr[mc];

                candidate.cmax = 0;
                for (int m = 1; m <= nbmachine; m++) {
                    if (candidate.tr[m] > candidate.cmax) candidate.cmax = candidate.tr[m];
                }

                // Heuristic score evaluation
                candidate.score = -(double)candidate.cmax + (rand() % 100) * 0.01;

                if (next_beam.size() < (size_t)kbest) {
                    next_beam.push(candidate);
                } else if (candidate.score > next_beam.top().score) {
                    next_beam.pop();
                    next_beam.push(candidate);
                }
            }
        }

        beam = next_beam;
        if (beam.empty()) break;
    }

    if (!beam.empty()) {
        best_makespan = beam.top().cmax;
    }
    return best_makespan;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        if (!resolve_instance_file(argv[1], instance_file)) {
            cerr << "Error: Cannot locate instance file for: " << argv[1] << endl;
            return 1;
        }
    }
    if (argc > 2) {
        kbest = atoi(argv[2]);
        if (kbest < 1) {
            cerr << "Error: beam_width K must be >= 1" << endl;
            return 1;
        }
    }

    cout << "  Instance File : " << instance_file << endl;
    cout << "  Beam Width (K): " << kbest << endl;

    lecture_donnees(instance_file);

    clock_t start_time = clock();
    srand(12345); // Fixed seed for reproducible BS-ICH search

    int best_makespan = solution_initial(0);

    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    cout << "  ===========================       : " << nbjob << endl;
    cout << "  Scheduled Operations : " << nbjob * nbmachine << endl;
    cout << "  Best Makespan (Cmax) : " << best_makespan << endl;
    cout << "  Execution Time       : " << elapsed << " seconds" << endl;

    return 0;
}
