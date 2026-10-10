
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <queue>
#include <random>
#include <fstream>
using namespace std;
#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <mpi.h>
#include <limits>
char exemple[] = "./la33.txt";
int kbest=5000;//5000 gave good result for la17/15/14
int b_cmax = 500000;

struct arc
{
    int opd, opf, type;
    int poid;
};

struct arcalter
{
    struct arc a, b;
    int num;
};

struct operation
{
    int num_op;
    int num_mc;
    int dure;
    int td, tf;
    int tdp, tfp;
    int num_j;
    int type;
    double ds;
    int ops;
    // int opp[hj]={0};
};

int nbjob, nbmachine, nboperation;
vector<operation> list_opr;
vector<int> opr_mach[20];
vector<arc> arc_fix;
vector<arcalter> list_paire;
int id_paire = 0;

/* pour ord*/
vector<int> dispo(31);
vector<int> tr(31);
vector<int> seq;
vector<int> ob[31];
vector<int> c;
int it = 0;
vector<int> pere(302);
vector<int> td(302);
vector<int> tf(302);

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};


using State = int;  // Example state representation (job-machine configuration)
using Action = int; // Example action representation (job index to be scheduled)
using QTable = std::unordered_map<std::pair<State, Action>, double, pair_hash>;

// Hyperparameters
// Exploration rate

const int max_episodes = 1000000;

double alpha = 0.1;
double gama = 0.1;
double epsilon = 1.0;

const double alpha_min = 0.01;
const double alpha_decay = 0.99995;
const double gamma_max = 0.5;
const double gamma_growth = 1.00001;
const double epsilon_min = 0.1;
const double epsilon_decay = 0.99995;
// Utility for pair hash to store Q-values

// Sample JobShopState with reward function for blocking job shop and makespan
class SchedulState
{
public:
    State state;
    bool is_complete;
    int makespan; // Tracks the makespan as scheduling progresses
    vector<State> all_states;
    vector<Action> all_actions;
    SchedulState(State initial_st, int cmax, vector<State> vs, vector<Action> va)
    {
        state = initial_st;
        if (initial_st < nbjob * nbmachine)
        {
            is_complete = false;
        }
        else
        {
            is_complete = true;
        }
        makespan = cmax;
        all_states = vs;
        all_actions = va;
    }

    State take_action(Action action)
    {
        // Simulate taking action and return the new state
        // Update 'state' with the result of action, e.g., scheduling a job
        // Here, we'll mock it as changing the state string and increasing makespan
        //  state += std::to_string(action);
        all_actions.push_back(action);
        all_states.push_back(state);
        // makespan += get_processing_time(action);  // Update makespan based on action
        return state;
    }

    int get_reward(Action action)
    {
        // Mock reward function: positive reward for non-blocking, negative for blocking
        if (dispo[action] != 0)
        {
            return 0; // Penalty for blocking
        }
        else
        {
            return 0; // Reward for successful scheduling
        }
    }

    int get_final_reward() const
    {
        // Final reward based on makespan (lower makespan, higher reward)
        if (makespan == 0)
            return 0; // Prevent division by zero
        return 1000000 / makespan;
    }

private:
    int get_processing_time(Action action)
    {
        // Example processing time for each action (could be based on actual job data)
        return 5 * (action + 1); // Mock processing time for demonstration
    }
};

class QLearningScheduler
{
public:
    QLearningScheduler() : rng(static_cast<unsigned>(std::time(0))) {}

    Action choose_action(const SchedulState &state, const std::vector<Action> &actions, QTable &q_table)
    {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon)
        {
            // Exploration: choose a random action
            std::uniform_int_distribution<int> action_dist(0, actions.size() - 1);
            return action_dist(rng); // actions[action_dist(rng)];
        }
        else
        {
            // Exploitation: choose the action with the highest Q-value
            return get_best_action(state, actions, q_table);
        }
    }

    void update_q_value(SchedulState &state, Action action, int reward, SchedulState &next_state, QTable &q_table, const std::vector<Action> &actions)
    {
        double max_q_next = q_table[{next_state.state, get_best_action(next_state, actions, q_table)}];
        double &q_value = q_table[{state.state, action}];
        //  cout << "\n in apdate Q, Action:"<< action;
        // Calculate target Q-value
        double target_q = reward + gama * max_q_next;

        // Calculate loss (squared difference)
        double loss = std::pow(target_q - q_value, 2);
        //    std::cout << "Loss: " << loss << std::endl;

        // Update Q-value using the standard Q-learning rule
        q_value += alpha * (target_q - q_value);
    }

    void apply_final_reward(SchedulState &state1, QTable &q_table)
    {
        //   cout<< "\nApply final reward based on makespan\n";
        int final_reward = state1.get_final_reward();

        // Update Q-values of the final state-action pair with the makespan-based reward

        /*   for (const auto& [state_action, q_value] : q_table) {
               if (state_action.first == state.state) {
                   q_table[state_action] += alpha * (final_reward - q_table[state_action]);
               }
           }*/

        for (int i = 0; i < state1.all_states.size(); i++)
        {
            State st = state1.all_states[i];
            Action ac = state1.all_actions[i];
            // if (state_action.first == state.state)
            {
                q_table[{st, ac}] += alpha * (final_reward - q_table[{st, ac}]);
            }
        }
    }

private:
    std::mt19937 rng;

    Action get_best_action(const SchedulState &state, const std::vector<Action> &actions, QTable &q_table)
    {
        // Find the action with the highest Q-value
        Action best_action = actions[0];
        double max_q_value = q_table[{state.state, actions[0]}];
        vector<int> index;
        index.push_back(0);
        int ind = 0;
        for (const auto &action : actions)
        {

            double q_value = q_table[{state.state, action}];
            if (q_value > max_q_value)
            {
                max_q_value = q_value;
                best_action = action;
                index.clear();
                index.push_back(ind);
            }
            else if (q_value == max_q_value)
            {
                index.push_back(ind);
            }
            ind++;
        }
        return index[rand() % index.size()];
    }
};

int solution_initial(int iteration, int myrank);

extern "C"

    int
    p1pv(vector<int> &a, int v)
{
    int ss = -1;
    if (a.size() > 0)
    {
        std::vector<int>::iterator res;
        res = std::find(a.begin(), a.end(), v);
        int a1 = std::distance(a.begin(), res);
        if (a1 != a.size())
            ss = a1;
    }
    return ss;
}
vector<int64_t> hash_path;
void lecture_donnees(void)
{
    FILE *fichier = NULL;
    fichier = fopen(exemple, "r+");
    int a = 0, b = 0;

    fscanf(fichier, "%d %d\n", &nbjob, &nbmachine);
    // printf("aaaaaaaaaaaaaaaa :a:%d b:%d \n",nbjob,nbmachine);
    nboperation = (nbjob * nbmachine) + 2;
    int i = 0, j = 0;

    // list_opr[(nbjob*nbmachine)+1].num_op=(nbjob*nbmachine)+1;
    operation o;
    o.dure = 0;
    o.num_op = 0;
    o.num_mc = 0;
    o.num_j = 0;

    list_opr.push_back(o);
    // int opr[nbjob*nbmachine+2][2];
    int idf_op = 1;
    int id[30] = {0};
    for (i = 1; i <= nbjob; i++)
    {
        for (j = 1; j <= nbmachine; j++)
        {
            if (j == nbmachine)
            {
                fscanf(fichier, "%d %d\n", &a, &b);
                // printf(" %d %d\n",a,b);
            }
            else
            {
                fscanf(fichier, "%d %d", &a, &b);
                // printf(" %d %d",a,b);
            }

            // opr[idf_op][0]=a;opr[idf_op][1]=b;

            operation p;
            p.dure = b;
            p.num_op = idf_op;
            p.num_mc = a + 1;
            p.num_j = i;
            list_opr.push_back(p);
            opr_mach[a + 1].push_back(idf_op);

            idf_op++;
        }
        j = 1;
    }
    operation p;
    p.num_op = nbjob * nbmachine + 1;
    p.dure = 0;
    o.num_mc = 0;
    o.num_j = 0;
    list_opr.push_back(p);
    // printf("\no1111111111111111111111pnnnn%d %d",list_opr[211].dure,list_opr[212].dure);
    int ia = 0;
    int lim = nbjob * nbmachine;

    for (int i1 = 1; i1 < lim; i1++)
    {
        // printf("\naaaaaaaaa %d %d", i1, lim);
        if ((i1 % nbmachine) != 0)
        {
            arc o;
            o.opd = i1;
            o.opf = i1 + 1;
            o.poid = list_opr[i1].dure;
            o.type = 0;
            arc_fix.push_back(o);
        }
    }

    for (int i = 0; i < nbjob; i++)
    {
        arc a;
        a.opd = 0;
        a.opf = i * nbmachine + 1;
        a.poid = 0;
        a.type = 0;
        arc_fix.push_back(a);
    }

    for (int i = 1; i <= nbjob; i++)
    {
        arc a;
        a.opf = nbjob * nbmachine + 1;
        a.opd = i * nbmachine;
        a.poid = 0;
        a.type = 0;
        arc_fix.push_back(a);
    }

    // id_arc_fix=ia-1;
    // printf("\no2222222222222222222222pnnnn%d %d",list_opr[211].dure,list_opr[212].dure);

    i = 1;
    j = 0;
    int k = 0;
    for (i = 1; i <= nbmachine; i++)
    {
        for (j = 0; j < nbjob; j++)
        {
            for (k = j + 1; k < nbjob; k++)
            {
                arcalter al1;

                al1.num = id_paire;
                int op1 = opr_mach[i][j];
                int op2 = opr_mach[i][k];
                //  printf("\n ....op  %d %d ",op1 ,op2);
                if (op1 % nbmachine != 0)
                {
                    al1.a.opd = op1 + 1;
                    al1.a.opf = op2;
                    al1.a.poid = 0;
                    al1.a.type = 1;
                }
                else
                {
                    al1.a.opd = op1;
                    al1.a.opf = op2;
                    al1.a.poid = list_opr[op1].dure;
                    al1.a.type = 1;
                }
                int op = op1;
                op1 = op2;
                op2 = op;
                // printf("\n ....op  %d %d %d",op1 ,op2,op);

                if (op1 % nbmachine != 0)
                {
                    al1.b.opd = op1 + 1;
                    al1.b.opf = op2;
                    al1.b.poid = 0;
                    al1.b.type = 1;
                }
                else
                {
                    al1.b.opd = op1;
                    al1.b.opf = op2;
                    al1.b.poid = list_opr[op1].dure;
                    al1.b.type = 1;
                }
                list_paire.push_back(al1);
                id_paire++;
            }
        }
    }
    id_paire--;

    // printf("arc 209:%d %d %d",arc_fix[209].opd,arc_fix[209].opf,arc_fix[209].poid);
    // printf("\n ..........................................%d  %d", nbmachine,nbjob);
    // printf("\n idddd %d ::",list_paire[5].num);
    // printf("\n verification: operation 1: %d %d %d",list_opr[0].num_op,list_opr[0].dure,list_opr[0].num_j);
    // printf("\n verification: operation nbmachine: %d %d %d",list_opr[1].num_mc,list_opr[1].dure,list_opr[1].num_op);
}

int p1p(vector<int> &a)
{
    int ss = -1;
    for (int j = 0; j < a.size(); j++)
        if (a[j] != 0)
        {
            ss = j;
            j = 1000;
        }
    return ss;
}

vector<int> ord_mc[20];

#include <vector>
#include <iostream>

struct Operation
{
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

struct Arc
{
    int opd, opf, type;
    int poid;
};

struct Arcalter
{
    Arc a, b;
    int num;
};

class JobShopState
{
public:
    std::vector<int> dispo;
    std::vector<int> tr;
    std::vector<int> seq;
    std::vector<int> ob[31];
    std::vector<int> c;
    std::vector<int> pere;
    std::vector<int> td;
    std::vector<int> tf;
    int cmax;
    double score;
    int64_t s_hash;
    // Constructor
    JobShopState()
    {
        // Initialize any required state variables if needed
    }

    // Save the current state from the given variables
    void save_state(

        const std::vector<int> &current_dispo, const std::vector<int> &current_tr,
        const std::vector<int> &current_seq, std::vector<int> current_ob[31],
        const std::vector<int> &current_c, const std::vector<int> &current_pere,
        const std::vector<int> &current_td, const std::vector<int> &current_tf, int_fast64_t hash_code)
    {
        s_hash = hash_code;
        dispo = current_dispo;
        tr = current_tr;
        seq = current_seq;
       /* for (int i = 0; i < 31; ++i)
        {
            if (current_ob[i].size() > nbjob)
                current_ob[i].clear(); //{  for (int i1 = 0; i1 < current_ob[i].size(); ++i1) if(current_ob[i][i1]>0&&current_ob[i][i1]<nboperation) ob[i].push_back(current_ob[i][i1]);
                                       //}
                                       // else{std::vector<int> tep;
                                       //  tep.push_back(0);

            ob[i] = current_ob[i]; //}
                                   //    cout<<"\n i:"<< i;
                                   //   cout<<"\n i:"<< i;cout<<"\n i:"<< i;cout<<"\n i:"<< i;
        }*/
        c = current_c;
        pere = current_pere;
        //memcpy(&pere, current_pere.data(), current_pere.size());
        td = current_td;
        tf = current_tf;
    }

    // Load the state into the given variables
    void load_state(
        std::vector<int> &current_dispo, std::vector<int> &current_tr,
        std::vector<int> &current_seq, std::vector<int> current_ob[31],
        std::vector<int> &current_c, std::vector<int> &current_pere,
        std::vector<int> &current_td, std::vector<int> &current_tf) const
    {
        current_dispo = dispo;
        current_tr = tr;
        current_seq = seq;
        /*for (int i = 0; i < 31; ++i)
        {
            current_ob[i] = ob[i];
        }*/
        current_c = c;
        current_pere = pere;
        current_td = td;
        current_tf = tf;
    }
};

int remaining(const JobShopState &state)
{
    int re = 0;
    vector<int> time_machines(20, 0);
    for (int i = 1; i <= nbmachine; i++)
    {
       time_machines[i]= state.tr[i];
    }

    for (const auto &op : list_opr) {
        if (std::find(state.seq.begin(), state.seq.end(), op.num_op) == state.seq.end()) {
            time_machines[op.num_mc] += op.dure;
        }
    }
    return re;
}

struct CompareTask
{
    bool operator()(const JobShopState &t1, const JobShopState &t2)
    {
      //  int re1=remaining(t1);
       // int re2=remaining(t2);
      {
        return t1.score < t2.score; // Max-Heap: worst candidate at the front
      }
      {

     // return (t1.cmax) >(t2.cmax); // Lower cmax means higher priority
      }//*/
     // return (re1) >(re2);
    }
};

int traiter(int cii)
{
    vector<int> afaire1;
    int ci = cii;
    // afaire[0]=ci;
    do
    {
        // printf("\ntraiter ci:%d",ci);
        int mci = list_opr[ci].num_mc;
        seq.push_back(ci);
        // le calcule de l'\E9valuation de la solution***********************************************
        if ((ci % nbmachine) != 1)
        {
            int opp = ci - 1;
            int mach_opp = list_opr[opp].num_mc;
            int cx = 0;

            // printf("\n pere[%d]:%d",ci,pere[ci]);
            if (tr[mci] <= tr[mach_opp])
            {
                cx = tr[mach_opp];
                pere[ci] = opp; // printf("\n pere[%d]:%d",ci,pere[ci]);
            }
            else
            {
                cx = tr[mci];
                int h = p1pv(ord_mc[mci], ci);
                int pp = -100;
                if (h > 0)
                    pp = ord_mc[mci][h - 1];
                /*prede[mci];*/ if (pp % nbmachine != 0)
                    pp++;

                pere[ci] = pp;
            }
            tr[mach_opp] = cx;
            tf[opp] = cx;
            td[ci] = cx;
            tr[mci] = cx + list_opr[ci].dure;
        }
        else
        {
            int h = p1pv(ord_mc[mci], ci);
            int pp = -100;
            if (h > 0)
            {
                pp = ord_mc[mci][h - 1];
                if (pp % nbmachine != 0)
                    pp++;
                pere[ci] = pp; // if(ci==26)  printf("\n pere[%d]:%d ::h:%d pp:%d",ci,pere[ci],h,pp);
            }
            else
            {
                pere[ci] = 0; // if(ci==26) printf("\n pere[%d]:%d ::h:%d pp:%d",ci,pere[ci],h,pp);
            }
            // if(pp%nbmachine!=0)pp++;
            td[ci] = tr[mci];
            tr[mci] = tr[mci] + list_opr[ci].dure;
        }
        // Fin calcule de l'\E9valuation*************************************************************
        // if(ob[mci].size()>0){int h1=p1pv(ob[mci],ci);
        // if(h1>0){printf("\nsupprision de ob[%d]:%d",mci,ob[mci][h1]);
        // ob[mci].erase(ob[mci].begin()+h1);}
        // }
        // res[nj]=seq[nj];

        dispo[mci] = ci; // prede[mci]=ci;
        // traiter(ci,0);

        int mo = ci % nbmachine; // printf("jcj:%d mo:%d",ci,mo);
        if (mo != 1)
        {
            int m = list_opr[ci - 1].num_mc;
            dispo[m] = 0;
            int jj1 = ob[m].size();
            // printf("jj1:%d",jj1);
            if (jj1 > 0)
            {
                int h = ob[m][0];
                afaire1.push_back(h);
                // printf("\nsupprision de ob[%d]:%d",mci,ob[m][0]);

                ob[m].erase(ob[m].begin());
            }
        }
        // traiter plus tard
        // pour le moment c'est moi meme
        if ((ci % nbmachine) != 0)
        {
            // if(p1pv(c,(ci+1),30)==-1)
            c.push_back(ci + 1);
            // res[1]=/*hhj*/ci;
            // res[2]=ci+1;
        }
        else { tf[ci] = tr[mci];
                 if (tf[ci] > td[nboperation - 1]) td[ nboperation - 1] = tf[ci];
           /* int jj = ob[mci].size();
            // int h=p1pv(ob,mci,nboperation);
            if (jj > 0)
            {
                int h = ob[mci][0];
                // if(p1pv(afaire1,h,50)==-1){
                afaire1.push_back(h);

                ob[mci].erase(ob[mci].begin());
                //}
            }
            else*/
            {
                dispo[mci] = 0;
            }
        }

        if (afaire1.size() == 0)
        {
            ci = -1;
        }
        else
        {
            ci = afaire1[afaire1.size() - 1];
            afaire1.pop_back();
        }
    } while (ci != -1);
    //}
    // printf("\nfin traiter\n");
    // printf("\nàààààààààààààààààà---fin----ààààààààààààààààààààààà");
    return 5;
}

// vector<int> td_it2 = td;

int traiter_swap(int cii, vector<int> &afaire)
{
    // printf("\nàààààààààààààààààà---debut----ààààààààààààààààààààààà");
    int ci = cii;
    int mci = list_opr[ci].num_mc;
    //int l = ob[mci].size();

    // list.push_back(add[a]);
    // add.erase(add.begin()+a);
    //int h1 = p1pv(ob[mci], ci); // printf("\n************** ci:%d mci:%d obmci:%d, h1:%d obmci[0]:%d",ci,mci,l,h1,ob[mci][0]);//printf("\n************** ci:%d mci:%d obmci:%d, h1:%d obmci[0]:%d",ci,mci,l,h1,ob[mci][0]);printf("\n************** ci:%d mci:%d obmci:%d, h1:%d obmci[0]:%d",ci,mci,l,h1,ob[mci][0]);printf("\n************** ci:%d mci:%d obmci:%d, h1:%d obmci[0]:%d",ci,mci,l,h1,ob[mci][0]);
    // if(a==ob[mci].size())h1=-1;
    // if(h1!=a)printf("\n -------------------------- a:%d h1:%d obsize:%d\n",a,h1,ob[mci].size());
    // if(h1+1==l){ob[mci].pop_back();}else{ob[mci][h1]=ob[mci][l-1];ob[mci].pop_back();}
    //ob[mci].erase(ob[mci].begin() + h1);

    seq.push_back(ci);
    // le calcule de l'\E9valuation de la solution***********************************************
    tr[mci] = tr[mci] + list_opr[ci].dure;
    // Fin calcule de l'\E9valuation*************************************************************
    // res[nj]=seq[nj];
    dispo[mci] = ci;
    // prede[mci]=ci;
    // traiter plus tard
    // pour le moment c'est moi meme
    if (ci % nbmachine != 0)
    {
        c.push_back(ci + 1);
    }
    else
    {
        /*int h = ob[mci].size();
        if (h > 0) // todo ad random
        {
            int hi = rand() % h;
            // if(p1pv(afaire,h,15)==-1){//pour verifier la répititions
            int succ = ob[mci][hi];
            ob[mci].erase(ob[mci].begin() + hi);
            afaire.push_back(succ);
            dispo[mci] = succ;
            tf[ci] = tr[mci];
            //}
        }
        else
        {*/
            dispo[mci] = 0;
        //}
    }
    int g = afaire.size(), rt = 0;
    if (g > 0)
        rt = afaire[0];
    // printf("\nàààààààààààààààààà---fin----ààààààààààààààààààààààà");
    return rt;
}

vector<int> b_seq;

clock_t tt1, tt2;


    int main(int argc, char **argv)
    {
    if (argc > 1) {
     if (argv[2]!=NULL) {  kbest = atoi(argv[2]);}
     if (argv[1]!=NULL) {
     exemple[0] = '\0';
    strcat(exemple, "./la");
    strcat(exemple, argv[1]);
    strcat(exemple, ".txt");}
    if (kbest < 1) {
        std::cerr << "Error: kbest must be greater than or equal to 1." << std::endl;
        return 1;
    }

    ifstream file(exemple);
    if (!file) {
        std::cerr << "Error: File " << exemple << " does not exist." << std::endl;
        return 1;
    }
    file.close();
}
    
    tt1 = clock();
        int size1 = 0;
        int global_best_cmax;
      
        lecture_donnees();


        MPI_Init(&argc, &argv);

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int local_best_cmax = b_cmax;
        if(world_rank==0) printf("exemple:%s k:%d ", exemple,kbest);
        srand(time(NULL)+world_rank);
        for (int i = 0; i < 500; i++)
        {
            int cm = solution_initial(i, world_rank);
            if (local_best_cmax > cm)
                local_best_cmax = cm;
            //printf("\nProcess %d - best solution:%d ", world_rank, local_best_cmax);

            
                    MPI_Reduce(&local_best_cmax, &global_best_cmax, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            printf("\niteration: %d Global best solution:%d ", i,global_best_cmax);
        } 
    b_cmax=global_best_cmax;
        //printf("\nProcess %d - best solution:%d ", world_rank, b_cmax);
        //printf("\nProcess %d - best solution:%d ", world_rank, local_best_cmax);
        }
        tt2 = clock();
        double elapsed_time = double(tt2 - tt1) / CLOCKS_PER_SEC;
        if (world_rank == 0)
        {
            printf("\nElapsed time: %.2f seconds\n", elapsed_time);
    }

        
      

        MPI_Finalize();

      //  printf("\nbest solution:%d ", b_cmax);
        return 0;
    }



vector<int> solinit_swap_new()
{
    vector<vector<int>> all_swap;
    vector<int> v;
    vector<int> possible_strats;

    int ss = -1;
    int f = -1;

    for (int i = 0; i < dispo.size(); i++)
    {
        f = -1;
        int op = dispo[i];
        int op1 = op + 1;
       // int h1 = p1pv(c, op1);
        //int h2 = p1pv(ob[list_opr[op1].num_mc], op1);
        vector<int> swap;
        if ((dispo[i] != 0) && dispo[list_opr[op1].num_mc]>0)
        {
            f = i;
            possible_strats.push_back(i);
        }

        if (f >= 0 && f < dispo.size())
        {

            swap.push_back(f);
            // printf("\n +++++++++ swap machine:%d dispo op:%d ",f,dispo[f]);
            int rt = 0;
            do
            {
                int suc = list_opr[dispo[f] + 1].num_mc;
                int op1 = dispo[suc] + 1;
                //int h2 = p1pv(ob[list_opr[op1].num_mc], op1);
                if (dispo[suc] != 0 and dispo[suc] <= nboperation && dispo[list_opr[op1].num_mc]>0)
                {
                    rt = p1pv(swap, suc);
                    //  printf("\n +++++++++ add swap mc:%d dispo op:%d rt:%d",suc,dispo[suc],rt);
                    if (rt == -1)
                    {
                        swap.push_back(suc);
                        f = suc;
                    }
                    else
                    {
                        for (int i = 0; i < rt; i++)
                        {
                            swap.erase(swap.begin());
                        }
                        // ss = 100;
                        //    i=1000000;
                    }
                }
                else
                {
                    // printf("\n +++++----++++ swap clear:%d dispo m:%d ",f,dispo[f]);
                    swap.clear();
                    rt = 0;
                }

            } while (rt == -1);
        }
        if (swap.size() >= 2)
            all_swap.push_back(swap);
    }
    int max_size = 0;
    int max_index = 0;
    for (int i = 0; i < all_swap.size(); i++) {
        if (all_swap[i].size() > max_size) {
            max_size = all_swap[i].size();
            max_index = i;
        }
    }
    if (max_size > 1) {
        return all_swap[max_index];
    } else {
        return v;
    }
   /* if (all_swap.size() > 1)
    { // cout << "\nall swaps size "<< all_swap.size()<<"";
        int ind = rand() % all_swap.size();
        return all_swap[ind];
    }
    else
    {
        return v;
    }*/
}

vector<int> solinit_swap()
{
    vector<vector<int>> all_swap;
    vector<int> v;
    vector<int> possible_strats;

    int ss = -1;
    int f = -1;

    for (int i = 0; i < dispo.size(); i++)
    {
        f = -1;
        int op = dispo[i];
        int op1 = op + 1;
        int h1 = p1pv(c, op1);
        int h2 = p1pv(ob[list_opr[op1].num_mc], op1);
        vector<int> swap;
        if ((dispo[i] != 0) && ob[i].size() > 0 && h1 < 0 && h2 >= 0)
        {
            f = i;
            possible_strats.push_back(i);
            // i = 300000;
            //    printf("\nh1 : %d",h1);
        }

        // f = possible_strats[0];

        // do{
        /*            std::random_device rd;  // Used as a non-deterministic seed
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> intDist(1, possible_strats.size());
        if(possible_strats.size()>0){
           int ind=  intDist(gen);
        f=possible_strats[ ind];
        possible_strats.erase(ind);
         //std::cout << intDist(gen) << ", |- " << possible_strats.size()<<"-|,";
        }
        */
        if (f >= 0 && f < dispo.size())
        {

            swap.push_back(f);
            // printf("\n +++++++++ swap machine:%d dispo op:%d ",f,dispo[f]);
            int rt = 0;
            do
            {
                int suc = list_opr[dispo[f] + 1].num_mc;
                int op1 = dispo[suc] + 1;
                int h2 = p1pv(ob[list_opr[op1].num_mc], op1);
                if (dispo[suc] != 0 and dispo[suc] <= nboperation && h2 >= 0)
                {
                    rt = p1pv(swap, suc);
                    //  printf("\n +++++++++ add swap mc:%d dispo op:%d rt:%d",suc,dispo[suc],rt);
                    if (rt == -1)
                    {
                        swap.push_back(suc);
                        f = suc;
                    }
                    else
                    {
                        for (int i = 0; i < rt; i++)
                        {
                            swap.erase(swap.begin());
                        }
                        // ss = 100;
                        //    i=1000000;
                    }
                }
                else
                {
                    // printf("\n +++++----++++ swap clear:%d dispo m:%d ",f,dispo[f]);
                    swap.clear();
                    rt = 0;
                }

            } while (rt == -1);
        }
        if (swap.size() >= 2)
            all_swap.push_back(swap);
    }

    // std::random_device rd;  // Used as a non-deterministic seed
    //      std::mt19937 gen(rd());
    //    std::uniform_int_distribution<> intDist(0, all_swap.size());

    /*if(all_swap.size()>1)if(all_swap[ind].size()>1){for (int i=0;i<all_swap[ind].size();i++){swap1.push_back(all_swap[ind][i]);
    cout << "\n swap op:"<<all_swap[ind][i];}
    ss=100;}*/
    //}    while (swap.size()>0||possible_strats.size()==0);
    //}
    //  for (int i=0;i<swap.size();i++){printf("sw:%d ",swap[i]);printf("sw:%d ",swap[i]);printf("sw:%d ",swap[i]);printf("sw:%d ",swap[i]);}
    if (all_swap.size() > 1)
    { // cout << "\nall swaps size "<< all_swap.size()<<"";
        int ind = rand() % all_swap.size();
        return all_swap[ind];
    }
    else
    {
        return v;
    }
}
vector<JobShopState> candidates;
int estimated_cost()
{
    int estimated = 0;
    for (int i = 0; i < nbmachine; i++)
    {
        int a = tr[i];
        for (int j = 0; j < ob[i].size(); j++)
        {
            a = a + list_opr[ob[i][j]].dure;
        }
        /*   for (int j=0;j<opr_mach[i].size();j++){
            a=a+list_opr[opr_mach[i][j]].dure;
        }*/
        if (a > estimated)
            estimated = a;
    }
    return estimated;
}
QTable qt;
vector<int> best_seq;

void in_element(vector<JobShopState>& top_elements, JobShopState current_element, int k)
{
    if (top_elements.size() < k) {
        top_elements.push_back(current_element);
        std::push_heap(top_elements.begin(), top_elements.end(), CompareTask{});
    } else {
        if (current_element.score < top_elements.front().score) {
            // Replace the worst element (largest score) in the heap
            std::pop_heap(top_elements.begin(), top_elements.end(), CompareTask{});
            top_elements.back() = current_element;
            std::push_heap(top_elements.begin(), top_elements.end(), CompareTask{});
        }
    }
}
void check_swap(){
    ////////////////////////////////////////////////////////////////////////////////
    vector<int> swapp;
    vector<int> afaire;
    int re = 0;
  //if(rand()%2==0)
   swapp = solinit_swap_new();

    if (swapp.size() > 1)
    {
        re = 100;
    }
    if (re > 0)
    {
        // le calcule de l'\E9valuation de la solution***********************************************
        int swt = 0; // sawp time
        int poo = 0; // swap operation with swt
        if (swapp.size() > 1)
        { // calculating the swap time (swt)
            for (int j = 0; j < swapp.size(); j++)
            {
                int m = swapp[j];
                if (m != 0)
                {
                    if (tr[m] > swt)
                    {
                        swt = tr[m];
                        poo = dispo[m];
                    }
                }
            }
            // pere[poo+1]=poo;
            poo++;
            // set swt for all swap machines
            for (int j = 0; j < swapp.size(); j++)
            {
                int m = swapp[j];
                if (m != 0)
                {
                    tr[m] = swt;
                }
            }

            // swapp now contains operations blocked
            for (int j = 0; j < swapp.size(); j++)
            {
                int mm = swapp[j];

                if (mm != 0)
                {
                    swapp[j] = dispo[mm] + 1;
                }
            }
            // updating pere vector

            for (int j = 0; j < swapp.size(); j++)
            {
                int op = swapp[j];
                int mpo = list_opr[op].num_mc;
                if (op != 0)
                {
                    if (op == poo)
                    {
                        pere[op] = poo - 1;
                    }
                    else
                    {
                        pere[op] = dispo[mpo] + 1;
                    }
                }
            }
            // traiter les operations de swap.
            for (int j = 0; j < swapp.size(); j++)
            {
                int cci = swapp[j];
                auto it = std::find(c.begin(), c.end(), cci);
                if (it != c.end()) {
                    c.erase(it);
                }
                //    printf("\n ci:%d swapp size:%ld",cci,swapp.size()) ;
                int mci = list_opr[cci].num_mc;
               // int l = ob[mci].size();
                //  printf("\n---------------- ci:%d mci:%d ci+1 m :%d",cci,mci,list_opr[cci+1].num_mc);
                traiter_swap(cci, afaire);
                swapp[j] = 0;
            }
            // traiter all operations that need to be traited afaire
           /* while (afaire.size() > 0)
            {
                //    printf("\n done afaire:%ld", afaire.size());
                int gh = afaire[afaire.size() - 1];
                afaire.pop_back();
                // if(p1pv(seq,gh)==-1)
                traiter(gh);
            }*/
        }
    }
    ////////////////////////////////////////////////////////////////////////////////
    
}
int solution_initial(int iteration, int myrank)
{ double elapsed_time=0;
    State initial_st = 0;
    vector<Action> va;
    std::vector<JobShopState> top_elements;
    top_elements.reserve(kbest); 
    vector<State> vs;
    SchedulState ch(initial_st, 0, vs, va);
    QLearningScheduler ql;
    int i0 = 0;
    candidates.clear();
    JobShopState savedState;
    JobShopState best_move;
  
    std::priority_queue<JobShopState, std::vector<JobShopState>, CompareTask> pq;
     //   vector<JobShopState> pq;
    
    int nb = 0;
    int y = 0;
    int gt = 0;
    int re = 0;
    int fh = 0;
    int res[100];
    // int i = threadIdx.x;
    seq.clear();
    c.clear();

    for (int i = 0; i < dispo.size(); i++)
    {
        dispo[i] = 0;
        ob[i].clear();
        tr[i] = 0;
    }

    for (int j = 0; j < nbjob; j++)
    {
        int aj = j * nbmachine + 1;
        c.push_back(aj);
    }
    int newop = 1;
    int best = 10000000;
    // if(i==0)
     savedState.save_state(dispo, tr, seq, ob, c, pere, td, tf, 0);
     candidates.push_back(savedState);
    while (newop< ((nbjob * nbmachine) + 10)  && candidates.size() > 0) //&&candidates.size() > 0 && newop < nbjob * nbmachine + 10 /*(c.size() > 0 && (fh < 10013)) || ((c.size() == 0) && (p1p(dispo) >= 0)*/)
    {
        newop++;
        //  cout << "\n"      << candidates.size() << " new iteration of the while loop \n";
      //    cout << "\n"<<candidates.size() <<" new iteration of the while loop \n"; cout << "\n"<<candidates.size() <<" new iteration of the while loop \n"; cout << "\n"<<candidates.size() <<" new iteration of the while loop \n"; cout << "\n"<<candidates.size() <<" new iteration of the while loop \n"; cout << "\n"<<candidates.size() <<" new iteration of the while loop \n"; cout << "\n"<<candidates.size() <<" new iteration of the while loop \n";
    for (int id_c = 0; id_c < candidates.size(); id_c++)
        {   
            clock_t start_time = clock();
            candidates[id_c].load_state(dispo, tr, seq, ob, c, pere, td, tf);
            clock_t end_time = clock();
            elapsed_time = elapsed_time+ double(end_time - start_time) / CLOCKS_PER_SEC;
          
           
          //candidates[id_c].load_state(dispo, tr, seq, ob, c, pere, td, tf);

        //    cout << "\n"
        //         << "iteration " << newop << " candidate " << id_c << " of " << candidates.size() << " \n";
//
        //    cout << seq.size() << " seq size \n";
        // if(ii0<candidates.size())if(candidates[ii0].c.size()>0)
        // 
        vector<int> ready_opp;
        for (int i = 0; i < c.size(); i++)
        {
            if (dispo[list_opr[c[i]].num_mc] == 0)
                ready_opp.push_back(c[i]);
        } 
            //  hash_path.push_back( candidates[id_c].s_hash);
            
            // re = solinit_swap(swapp);
            nb = ready_opp.size(); // c.size();
            int t = 0, mt = 0;
            int cnd = 0;
            int bo = 0;
            // int mtb = -1;

               // printf("\n cnd :%d bo:%d nb:%d mtb:%d",cnd, bo,nb);
            // std::random_device rd; // Used as a non-deterministic seed
            // std::mt19937 gen(rd());
            //  std::uniform_int_distribution<> intDist(0, c.size());
            // if(possible.size()==0) nb=0;
            
            if (nb > 0)
            {
               
                int ii0 = rand() % nb;
               // if(newop<4)
               // {ii0=myrank%nb;}
           /*  if(iteration%3==0)
           { int max_elements = 0;
            for (int i = 0; i < ready_opp.size(); i++) {
                int mc = list_opr[ready_opp[i]].num_mc;
                int elements = std::count_if(ready_opp.begin(), ready_opp.end(), [mc](int op) {
                    return list_opr[op].num_mc == mc;
                });
                if (elements > max_elements) {
                    max_elements = elements;
                    ii0 = i;
                }
            }} else 
            if(iteration%3==1)
            { int max_elements = 0;
             for (int i = 0; i < ready_opp.size(); i++) {
                 int mc = list_opr[ready_opp[i]].num_mc;
                 int elements = std::count_if(ready_opp.begin(), ready_opp.end(), [mc](int op) {
                     return list_opr[op].num_mc == mc;
                 });
                 if (elements < max_elements) {
                     max_elements = elements;
                     ii0 = i;
                 }
             }}*/
           int  mtb=  list_opr[ready_opp[ii0]].num_mc;
                int in = 1; // int mtb=list_opr[c[rand()%c.size()]].num_mc;
                 for (int it = 0; it < nb; it++)
                if(list_opr[ready_opp[it]].num_mc==mtb)
                {candidates[id_c].load_state(dispo, tr, seq, ob, c, pere, td, tf);
                    int ac_ = /*c*/ ready_opp[it];
                    { // l=10000;
                        in = 1;
                        // 
                        gt = it;
                        int ci = /*c*/ ready_opp[gt];
                        // c.erase(c.begin() + gt);
                        auto itt = std::find(c.begin(), c.end(), ci);
                        if (itt != c.end())
                        {
                            c.erase(itt);
                        }

                        int mci = list_opr[ci].num_mc;

                        if (dispo[mci] == 0)
                        {
                            traiter(ci);
                        }
                        else
                        {

                            /*ob[mci].push_back(ci);*/
                        }

                        check_swap();
                        std::vector<int>::iterator result;
                        result = std::max_element(tr.begin(), tr.end());
                        int a = std::distance(tr.begin(), result);
                        int64_t p_hash = 0; 
                        
                        clock_t start_time = clock();
                        JobShopState tmp;
                        tmp.save_state(dispo, tr, seq, ob, c, pere, td, tf, p_hash);
                        tmp.cmax = tr[a];
                        
                        // Add random tie-breaker for diversity
                        double noise = (double)(rand() % 1000) / 1000000.0;
                        tmp.score = ((double)tmp.cmax / seq.size()) + noise;

                        // Branch and Bound: Prune if this branch is already worse than the global best
                        if (tmp.cmax < b_cmax) {
                            in_element(top_elements, tmp, kbest);
                        }
                        
                        clock_t end_time = clock();
                        elapsed_time = elapsed_time + double(end_time - start_time) / CLOCKS_PER_SEC;
                        
                        if (tr[a] < best && seq.size() == nbjob * nbmachine)
                        { 
                            best = tr[a];
                        }
                        // savedState.load_state( dispo, tr, seq, ob, c, pere, td, tf );
                        // }*/

                        // cout<<"it:"<<id_c << "inside amd saved new config\n";
                       // SchedulState nch(ch.state + 1, tr[a], ch.all_states, ch.all_actions);

                       /* ql.update_q_value(ch, ac_, rwd, nch, qt, c);
                        if (seq.size() >= nbjob * nbmachine)
                        {
                            ql.apply_final_reward(nch, qt);
                            /*    for (int i0=0;i0<nboperation-1;i0++){
                             cout<<"\n";
                             for (int i1=1;i1<nboperation;i1++){
                             cout << " "<< qt[{i0, i1}];
                             }
                         }*/
                      //  }                        ch = nch;
                    }
                    // if(in ==0){ pq.emplace(candidates[id_c]);  }

                    // best_move.load_state(dispo, tr, seq, ob, c, pere, td, tf );
                }
            }

            else
            {   clock_t start_time = clock();
                candidates[id_c].load_state(dispo, tr, seq, ob, c, pere, td, tf);
                clock_t end_time = clock();
                elapsed_time = elapsed_time+ double(end_time - start_time) / CLOCKS_PER_SEC;
                check_swap();
                std::vector<int>::iterator result;
                result = std::max_element(tr.begin(), tr.end());
                int a = std::distance(tr.begin(), result);
                // printf("\n solution move i:%d cmax:%d",it,tr[a]);
                int64_t p_hash = 0; 
                
                start_time = clock();
                JobShopState tmp;
                tmp.save_state(dispo, tr, seq, ob, c, pere, td, tf, p_hash);
                tmp.cmax = tr[a];
                
                double noise = (double)(rand() % 1000) / 1000000.0;
                tmp.score = ((double)tmp.cmax / seq.size()) + noise;

                if (tmp.cmax < b_cmax) {
                    in_element(top_elements, tmp, kbest);
                }
                
                end_time = clock();
                elapsed_time = elapsed_time + double(end_time - start_time) / CLOCKS_PER_SEC;
            }
        }  
        //std::cout << pq.size() << "Tasks in the priority queue (by increasing priority):\n";
        //  candidates.clear();

         int k = 0;
         int ncb = kbest; // Define a vector of doubles and a target value
         vector<int> values = {0};
         double target = 3.5;

         // Count the occurrences of the target value in the
         // vector
   
         // Optional: Sort the top elements in descending order
   // std::sort(top_elements.begin(), top_elements.end(), CompareTask{});
         candidates.clear();
         candidates=top_elements;
         top_elements.clear();
         clock_t start_time = clock();
        /* while (!pq.empty())
         {
           if (k < ncb)
                if(pq.top().seq.size()<nbjob*nbmachine)
                 {

                     if(k<ncb)
                     {
                     candidates.push_back(pq.top());
                     k++;
                     }
                 }
        //   pq.erase(pq.begin()+id);
        if(k>=ncb) {pq = std::priority_queue<JobShopState, std::vector<JobShopState>, CompareTask>();}
        else    if(pq.empty()==false)   pq.pop();

    }  */   
    clock_t end_time = clock();
    elapsed_time = elapsed_time+ double(end_time - start_time) / CLOCKS_PER_SEC;//estimated_cost();//
    }  
    
   /* if(myrank==0){std::cout << "Time consumed by load_state: " << elapsed_time << " seconds\n";
    printf("\n solution cmax:%d", best);}*/
    best_seq = seq;
    /*  if (b_cmax > tr[a]) {
          b_seq = seq;
          b_cmax = tr[a];
      }*/
    return best;
}