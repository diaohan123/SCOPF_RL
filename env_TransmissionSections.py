from ast import get_docstring
from logging import exception
from os import getenv
from gym import spaces, core
from numpy.core.fromnumeric import ndim
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False
# from gym_psops.envs import psops
# from gym_psops.envs.psops import Py_PSOPS
from envs.psops.py_psops import Py_PSOPS
import numpy as np
import ray
import time

# core.Env是gym的环境基类,自定义的环境就是根据自己的需要重写其中的方法；
# 必须要重写的方法有:
# __init__()：构造函数
# reset()：初始化环境
# step()：环境动作,即环境对agent的反馈
# render()：如果要进行可视化则实现


class sections_Env(core.Env):
    def __init__(
        self,
        flg=0,
        rng=None,
        sys_no=-1,
        act_gen_v=None,
        act_gen_p=None,
        sampler="stepwise",
        static_check="all",
        observation_type="minimum",  # minimum state, all state
        action_type="absolute",  # absolute action, or delta action
        reward_type="unstabletime",  # maximum rotor angle, TSI, unstable duration
        check_voltage=True,
        check_slack=True,
        upper_load=1.2,
        lower_load=0.7,
        criterion=180.0,
        fault_set=[[26, 0, 0.1], [26, 100, 0.1]],
        sections=[
            [["BUS-16", "BUS-15"], ["BUS-16", "BUS-17"]],
            # [["BUS-1","BUS-39"],["BUS-6","BUS-7"]],
            # [["BUS-2","BUS-1"],["BUS-2","BUS-3"],["BUS-26","BUS-27"]],
        ],
        disturbance_set=[],
        co_reward=2000.0,
    ):
        
        # flg and system numer
        self.__flg = flg
        self.__sys = sys_no
        # psops api
        self.psops = Py_PSOPS(self.__flg)
        api = self.psops
        self.co_static_violation = -100.0
        self.neg_mid = -500.0
        self.neg_max = -1000.0
        # cost coefficients
        self.ck1 = 0.2
        self.ck2 = 30.
        self.ck3 = 100.
        # max cost
        g_max = api.get_generator_all_pmax()
        self.max_cost = sum(self.ck1 + self.ck2 * g_max + self.ck3 * g_max * g_max)
        # settings
        self.sampler = sampler
        assert (
            self.sampler == "stepwise" or self.sampler == "simple"
        ), f"Unknown sampler type: {self.sampler}. Please check. "
        self.static_check = static_check
        assert (
            self.static_check == "all"
            or self.static_check == "important"
            or self.static_check == "none"
        ), f"Unknown static check type: {self.static_check}. Please check."
        self.observation_type = observation_type
        assert (
            self.observation_type == "minimum" or self.observation_type == "all"
        ), f"Unknown observation type: {self.observation_type}. Please check."
        self.action_type = action_type
        assert (
            self.action_type == "absolute" or self.action_type == "delta"
        ), f"Unknown action type: {self.action_type}. Please check."
        self.reward_type = reward_type
        assert (
            self.reward_type == "unstabletime"
            or self.reward_type == "maxrotordiff"
            or self.reward_type == "tsi"
        ), f"Unknown reward type: {self.reward_type}. Please check."
        self.check_voltage = check_voltage
        self.check_slack = check_slack
        self.criterion = criterion
        self.co_reward = co_reward
        
        ############# transmission set [i,j] (power from i to j) #############################################################################
        self.sections = sections
        self.sections_pmax = np.array([6*len(s) for s in self.sections])
        self.sections_l_no = self.get_sections_line_no()
        self.area_info = self.get_area()
        self.sections_neighbor = self.get_sections_neighbor()
        self.area_neighbor=self.get_area_neighbor()
        self.sections_info = [
            {
                "line_with_bus": self.sections[i],
                "line_no": self.sections_l_no[i],
                "neighbor_area": self.sections_neighbor[i],
                "pmax": self.sections_pmax[i],
            }
            for i in range(len(self.sections))
        ]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~灵魂之子，浇给~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
        # load limit
        self.load_lower_limit = upper_load
        self.load_upper_limit = lower_load



        # random generator
        self.set_random_state(np.random.default_rng() if rng is None else rng)
        # save original state
        self.__acline_connectivity = api.get_network_acline_all_connectivity()
        self.__generator_connectivity = api.get_network_generator_all_connectivity()
        self.__generator_v_set = api.get_generator_all_v_set()
        self.__generator_p_set = api.get_generator_all_p_set()
        self.__load_p_set = api.get_load_all_p_set()
        self.__load_q_set = api.get_load_all_q_set()

        # observation, V, theta, PG, QG, Pl, Ql
        self.__RADIAN = 180.0 / 3.1415926535897932384626433832795
        
  
        ob_bus_vmax = api.get_bus_all_vmax()
        ob_bus_vmin = api.get_bus_all_vmin()
        ob_generator_pmax = api.get_generator_all_pmax()
        ob_generator_pmin = api.get_generator_all_pmin()
        ob_generator_qmax = api.get_generator_all_qmax()
        ob_generator_qmin = api.get_generator_all_qmin()
        ob_load_pmax = api.get_load_all_p_set() * self.load_upper_limit
        ob_load_pmin = api.get_load_all_p_set() * self.load_lower_limit
        ob_load_qmax = api.get_load_all_q_set() * self.load_upper_limit
        ob_load_qmin = api.get_load_all_q_set() * self.load_lower_limit
        ob_section_pmax=self.sections_pmax
        ob_section_pmin=-ob_section_pmax

        lower = np.concatenate(
            (
                ob_bus_vmin,
                ob_generator_pmin,
                ob_generator_qmin,
                ob_load_pmin,
                ob_load_qmin,
                # ob_section_pmin,
            )
        )

        upper = np.concatenate(
            (
                ob_bus_vmax,
                ob_generator_pmax,
                ob_generator_qmax,
                ob_load_pmax,
                ob_load_qmax,
                # ob_section_pmax,
            )
        )
        idx = lower > upper
        lower[idx], upper[idx] = upper[idx], lower[idx]
        assert True not in (
            upper < lower
        ), "observation upper is smaller than lower, please check"
        self.__centralOb = 0.5 * (lower + upper)
        self.__deltaOb = 0.5 * (upper - lower)
        if self.observation_type == 'minimum':
            self.ob_no=range(len(lower)-self.psops.get_load_number()*2-len(self.sections),len(lower))
        elif self.observation_type == 'all':
            self.ob_no=range(len(lower))
        self.observation_space = spaces.Box(low=lower[self.ob_no], high=upper[self.ob_no])
        self.state_space = spaces.Box(low=lower, high=upper)
        self.state = self.state_space.sample()

        self.important_state_idx = np.arange(api.get_bus_number()+2*api.get_generator_number())
        # action, gen_V, gen_P
        self.__ctrl_v_gen = (
            np.arange(api.get_generator_number()) if act_gen_v is None else act_gen_v
        )
        self.__ctrl_p_gen = (
            api.get_generator_all_ctrl() if act_gen_p is None else act_gen_p
        )

        # max step
        self._max_episode_steps = 20
        # max cost
        g_max = api.get_generator_all_pmax()
        self.__max_cost = sum(100.0 * g_max * g_max + 30.0 * g_max + 0.2)

        # 动作空间
        # self.__originCtrl = self.get_ctrl()
        # self.__curCtrl = self.__originCtrl.copy()
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.sections),))

        # 灵敏度矩阵
        self.Sdg,self.Rgg_i,self.Gkg=self.cal_sensitivity_matrix()
        
        # anticipated contingency set
        fault_set=list()
        # if api.get_bus_number() == 39:
        #     fault_set = [[26, 0, 0.1], [26, 0, 0.1]]
        # elif api.get_bus_number() == 710:
        #     fault_set = [[100, 0, 0.1], [100, 100, 0.1]]
        for section in self.sections_l_no:
            for l in section:
                fault_set.append([l,0,0.1])
                fault_set.append([l,100,0.1])
        self.anticipated_fault_set = fault_set
        self.anticipated_disturbance_set = disturbance_set
        
        # dynamic constraints
        if self.reward_type == "unstabletime":
            self.c_fault = (
                (self.neg_mid - self.neg_max)
                / (
                    len(self.anticipated_fault_set)
                    + len(self.anticipated_disturbance_set)
                )
                / ((api.get_info_ts_max_step() - 1) * api.get_info_ts_delta_t())
            )
        elif self.reward_type == "maxrotordiff":
            self.c_fault = (
                (self.neg_mid - self.neg_max)
                / (
                    len(self.anticipated_fault_set)
                    + len(self.anticipated_disturbance_set)
                )
                / 500.0
            )
        elif self.reward_type == "tsi":
            self.c_fault = (self.neg_mid - self.neg_max) / (
                len(self.anticipated_fault_set) + len(self.anticipated_disturbance_set)
            )
        if self.sampler == "stepwise":
            self.pf_sampler = api.get_power_flow_sample_stepwise
        elif self.sampler == "simple":
            self.pf_sampler = api.get_power_flow_sample_simple_random

    def set_flg(self, flg):
        self.__flg = flg
        self.psops.set_flg(flg)

    def set_random_state(self, rng):
        self.__rng = rng
        self.psops.set_random_state(self.__rng)

    def seed(self, sd):
        self.__rng = np.random.default_rng(sd)
        self.psops.set_random_state(self.__rng)

    def get_psops(self):
        return self.psops
    
    def get_ctrl_gen(self):
        return np.concatenate([self.__ctrl_v_gen,self.__ctrl_p_gen])

    def set_back_to_origin_state(self):
        api = self.psops
        api.set_network_acline_all_connectivity(self.__acline_connectivity)
        api.set_network_generator_all_connectivity(self.__generator_connectivity)
        api.set_generator_all_v_set(self.__generator_v_set)
        api.set_generator_all_p_set(self.__generator_p_set)
        api.set_load_all_p_set(self.__load_p_set)
        api.set_load_all_q_set(self.__load_q_set)

    def get_sample(self):
        api=self.psops
        gen_v=api.get_generator_all_v_set(self.__ctrl_v_gen)
        gen_p=api.get_generator_all_p_set(self.__ctrl_p_gen)
        load_p=api.get_load_all_p_set()
        load_q=api.get_load_all_q_set()
        return np.concatenate([gen_v,gen_p,load_p,load_q])
        
    def get_generator_setting(self):
        api=self.psops
        gen_v=api.get_generator_all_v_set()
        gen_p=api.get_generator_all_p_set()       
        return np.concatenate([gen_v,gen_p])
    
    def get_sections_line_no(self):
        api = self.psops
        sections_l_no = list()
        for section in self.sections:
            l_no = list()
            for n, line in enumerate(section):
                [i, j] = line
                acline_no = np.array(
                    np.where(
                        (api.get_acline_all_i_name() == i)
                        & (api.get_acline_all_j_name() == j)
                    )
                ).reshape(-1)
                assert acline_no.size == 0 or acline_no.size == 1
                if acline_no.size == 0:  # same direction
                    acline_no = np.array(
                        np.where(
                            (api.get_acline_all_i_name() == j)
                            & (api.get_acline_all_j_name() == i)
                        )
                    ).reshape(-1)
                l_no.append(acline_no[0])
            sections_l_no.append(l_no)
        return sections_l_no

    def get_area(self):
        api = self.psops
        for section in self.sections_l_no:
            api.set_network_acline_all_connectivity(
                cmarks=np.full(len(section), False), acline_list=section
            )
        n_area = api.get_network_n_acsystem_check_connectivity()
        area = list()
        # print(n_area)
        for i in range(n_area):
            bus_area = api.get_network_bus_all_connectivity_flag()
            # print(bus_area)
            gen_area = bus_area[api.get_generator_all_bus_no()]
            load_area = bus_area[api.get_load_all_bus_no()]
            area_info = {
                "bus": np.array(np.where(bus_area == i)).reshape(-1),
                "gen": np.array(np.where(gen_area == i)).reshape(-1),
                "load": np.array(np.where(load_area == i)).reshape(-1),
            }
            area.append(area_info)
        api.set_network_topology_original()
        return area

    def get_sections_neighbor(self):
        api = self.psops
        sections_neighbor = list()
        for section in self.sections:
            bus_i_name = section[0][0]
            bus_i_no = api.get_bus_no(bus_i_name)
            bus_j_name = section[0][1]
            bus_j_no = api.get_bus_no(bus_j_name)
            sys_no = [0, 0]

            for area_no, area in enumerate(self.area_info):
                # print(area)
                # print(bus_i_no in area)
                if bus_i_no in area["bus"]:
                    sys_no[0] = area_no
                if bus_j_no in area["bus"]:
                    sys_no[1] = area_no
            sections_neighbor.append(sys_no)
        return sections_neighbor
    
    def get_area_neighbor(self):
        area_neighbor=list()
        for area_no, area in enumerate(self.area_info):
            area_sections=list()
            directions=list()
            for section_no,neighbor in enumerate(self.sections_neighbor):
                if neighbor[0]==area_no:
                    area_sections.append(section_no)
                    directions.append(1)
                if neighbor[1]==area_no:
                    area_sections.append(section_no)
                    directions.append(-1)
            area_neighbor.append({'section':area_sections,'direction':directions})
        return area_neighbor
                    
            
    def cal_sections_power(self):
        api = self.psops
        if api.cal_power_flow_basic_nr()<0:
            return np.zeros(len(self.sections))
        p_sections = list()
        for ns, section in enumerate(self.sections):
            p_section = list()
            for nl, line in enumerate(section):
                [i, j] = line
                line_no = self.sections_l_no[ns][nl]
                p = 0
                if api.get_acline_i_name(line_no) == i:  # same direction
                    p = api.get_acline_lf_result(acline_no=line_no)[0]
                if api.get_acline_i_name(line_no) == j:  # different direction
                    acline_no = np.array(
                        np.where(
                            (api.get_acline_all_i_name() == j)
                            & (api.get_acline_all_j_name() == i)
                        )
                    ).reshape(-1)
                    p = api.get_acline_lf_result(acline_no=line_no)[2]
                p_section.append(p)
            p_sections.append(sum(p_section))
        return np.array(p_sections)

    
    def cal_sensitivity_matrix(self):
        api=self.psops
        # Vg-Vd灵敏度和Vg-Qg灵敏度计算
        G,B=api.get_network_admittance_matrix_full()
        g=api.get_generator_all_bus_no()
        d=np.setdiff1d(api.get_load_all_bus_no(),g)
        Ldd=B[d,:][:,d]
        Ldg=B[d,:][:,g]
        Lgd=B[g,:][:,d]
        Lgg=B[d,:][:,g]
        R=-np.matrix(B).I
        Rgg=R[g,:][:,g]
        Rgg_i=Rgg.I
        Sdg=-np.matrix(Ldd).I*Ldg    
        
        #发电机出力分布因子计算
        slack_bus=api.get_generator_bus_no(api.get_generator_all_slack()[0])
        xk=np.array(api.get_acline_all_reactance())
        xt=np.array(api.get_transformer_all_real_reactance())
        y=1/np.concatenate([xk,xt])
        M=np.zeros((api.get_bus_number(),api.get_acline_number()+api.get_transformer_number()))
        for line in range(api.get_acline_number()):
            M[[api.get_acline_i_no(line),api.get_acline_j_no(line)],line]=[1,-1]
        for trans in range(api.get_transformer_number()):
            M[[api.get_transformer_i_no(trans),api.get_transformer_j_no(trans)],trans+api.get_acline_number()]=[1,-1]
        Y=np.diag(y)
        M=np.matrix(M)
        Y0=np.dot(M,np.dot(Y,M.T))
        Y=np.delete(np.delete(Y0, slack_bus, axis=0), slack_bus, axis=1)
        X=Y.I
        xk=np.array(api.get_acline_all_reactance())
        slack_bus=api.get_generator_bus_no(api.get_generator_all_slack()[0])
        e=np.zeros((api.get_bus_number(),api.get_generator_number()))
        for i,g in enumerate(self.__ctrl_p_gen):
            e[api.get_generator_bus_no(g),i]=1
        e=np.delete(e,slack_bus,axis=0)
        Xg=np.dot(X,e)
        Gkg=np.zeros((len(self.sections_info),api.get_generator_number()))
        for section_i, section in enumerate(self.sections_info):            
            for l, l_no in zip(section['line_with_bus'],section['line_no']):
                M=np.zeros(api.get_bus_number())
                i=api.get_bus_no(l[0])
                j=api.get_bus_no(l[1])
                M[[i,j]]=[1,-1]
                M=np.delete(M,slack_bus)                
                Xkg=np.dot(M.reshape(1,-1),Xg)
                Gkg[section_i]+=np.array(Xkg/xk[l_no]).reshape(-1)
        Gkg[np.where(np.abs(Gkg)<1e-9)]=0
        return Sdg, Rgg_i, Gkg
    
    def adjust_sensitivity(self):
        api=self.psops
        g=api.get_generator_all_bus_no()
        d=np.setdiff1d(api.get_load_all_bus_no(),g)
        Sdg=self.Sdg
        Rgg_i=self.Rgg_i
        Qg=api.get_generator_all_lf_result()[:,1]
        Vg=api.get_generator_all_v_set()
        Vd=api.get_bus_all_lf_result()[d,0]

        Qg_max=api.get_generator_all_qmax()
        Qg_max[api.get_generator_all_slack()]
        Qg_min=api.get_generator_all_qmin()
        Qg_mid=(Qg_max+Qg_min)/2
        Vd_max=api.get_bus_all_vmax()[d]
        Vd_min=api.get_bus_all_vmin()[d]
        Vd_mid=(Vd_max+Vd_min)/2
        Vg_max=api.get_generator_all_vmax()
        Vg_min=api.get_generator_all_vmin()

        P=matrix(np.eye(len(g)))
        
        # q=np.dot(2*Vd-2*Vd_mid,Sdg)
        q=np.zeros(len(g))

        # q=np.dot(2*Qg-2*Qg_mid,Rgg_i)
        # q=np.dot(np.concatenate([2*Vd-2*Vd_mid,2*Qg-2*Qg_mid]),np.concatenate([Sdg,Rgg_i]))
        q=matrix(np.array(q).reshape(-1).astype(np.float64))
        G=matrix(np.concatenate((Sdg,-Sdg, Rgg_i, -Rgg_i ,P,-P)).astype(np.float64))
        h=matrix(np.concatenate([Vd_max-Vd, Vd-Vd_min, Qg_max-Qg,Qg-Qg_min, Vg_max-Vg, Vg-Vg_min]).astype(np.float64))    
        try: 
            s=solvers.qp(P,q,G,h)
            dVg=np.array(s['x']).reshape(-1)
            api.set_generator_all_v_set(Vg+dVg)
        except Exception as e:
            print('电压灵敏度优化失败')
    
    def adjust_sections_power_sensitivity(self, p_set, goal='cost'):
        api=self.psops
        # 优化   
        Gkg=self.Gkg     
        p_now=self.cal_sections_power()
        it=0
        while 1:
            p_max=np.abs(p_set)
            p_min=-p_max
            Pg=api.get_generator_all_lf_result()[:,0]
            Pg_max=api.get_generator_all_pmax()
            Pg_max[api.get_generator_all_slack()]*=0.95
            Pg_min=api.get_generator_all_pmin()
            # print(f'Pg:{Pg}')
            if goal=='cost':
                P=matrix(self.ck3*np.eye(api.get_generator_number()))
                q=matrix(np.array(2*self.ck3*Pg+self.ck2).astype(np.float64))
            else:
                P=matrix(np.eye(api.get_generator_number()))
                q=matrix(np.zeros(api.get_generator_number()))
            # G=matrix(np.concatenate([np.eye(api.get_generator_number()),-np.eye(api.get_generator_number())]).astype(np.float64))
            # h=matrix(np.concatenate([Pg_max-Pg, Pg-Pg_min]).astype(np.float64))
            # A=matrix(Gkg)
            # print(f'A:{np.concatenate([Gkg,np.ones((1,api.get_generator_number()))])}')
            # b=matrix(p_set-p_now)
            # print(f'b:{np.concatenate([p_set-p_now,[0]])}')
            G=matrix(np.concatenate([np.eye(api.get_generator_number()),-np.eye(api.get_generator_number())]).astype(np.float64))
            h=matrix(np.concatenate([Pg_max-Pg, Pg-Pg_min]).astype(np.float64))
            A=matrix(np.concatenate([Gkg,np.ones((1,api.get_generator_number()))]))
            b=matrix(np.concatenate([p_set-p_now,np.zeros((1,))]))
            try: 
                s=solvers.qp(P,q,G,h,A,b)
                if s['status']=='optimal':
                    dPg=np.array(s['x']).reshape(-1)
                    api.set_generator_all_p_set(pset_array=Pg+dPg)
                    # print(f'dPg:{dPg}')
                else:
                    raise Exception
            except Exception as e:
                print('功率灵敏度优化失败')
                return 0
            it+=1
            if np.linalg.norm(p_now-p_set,ord=np.inf)>0.01 or it>10:
                break
        return 1
    
    def adjust_propotional(self, ctrl_gen_no, gen_no, p_set, p_now):
        api = self.psops
        dP_sum=p_set-p_now
        # ctrl_gen_no=gen_no  
        full_gen=[]    
        while np.abs(dP_sum) > 1e-5:
            # print(dP_sum)
            # print(ctrl_gen_no)
            ctrl_gen_no=list(set(ctrl_gen_no)-set(full_gen))
            if len(ctrl_gen_no)==0:
                break
            dP_avr=dP_sum/len(ctrl_gen_no)
            for g in ctrl_gen_no:
                P=api.get_generator_lf_result(g)[0]
                Pmax=api.get_generator_pmax(g)
                Pmin=api.get_generator_pmin(g)
                if P+dP_avr>Pmin and P+dP_avr<Pmax:
                    api.set_generator_p_set(pset=P+dP_avr,generator_no=g)
                    dP_sum-=dP_avr
                elif P+dP_avr<=Pmin:
                    api.set_generator_p_set(pset=Pmin,generator_no=g)
                    dP_sum-=Pmin-P
                    full_gen.append(g)
                else:
                    api.set_generator_p_set(pset=Pmax,generator_no=g)
                    dP_sum-=Pmax-P
                    full_gen.append(g)

    def adjust_areas(self,p_sections_set):
        p_sections=self.cal_sections_power()
        for area,neighbor in zip(self.area_info,self.area_neighbor):
            p_set=np.sum(p_sections_set[neighbor["section"]]*neighbor["direction"])
            p_now=np.sum(p_sections[neighbor["section"]]*neighbor["direction"])
            gen=area["gen"]
            ctrl_gen=np.intersect1d(gen,self.__ctrl_p_gen)
            load=area["load"]
            self.adjust_propotional(ctrl_gen_no=ctrl_gen, gen_no=gen, p_set=p_set,p_now=p_now)
            
 
    def set_start(self, sample):
        api=self.psops
        # gen_v=gen_set[:api.get_generator_number()]
        # gen_p=gen_set[api.get_generator_number():]
        # api.set_generator_all_v_set(gen_v)
        # api.set_generator_all_p_set(gen_p)
        api.set_power_flow_initiation(sample)
        obs, _ = self._get_observation()   
        # print(api.get_load_all_p_set())             
        return obs    
    
    def set_random_start(self):
        self.pf_sampler(num=1,
                load_max=self.load_upper_limit,
                load_min=self.load_lower_limit,
                check_voltage=self.check_voltage,
                check_slack=self.check_slack
                )

        obs, _ = self._get_observation()        
        
        return obs

    def set_insecure_start(self):
        while 1:
            self.pf_sampler(
                num=1,
                load_max=self.load_upper_limit,
                load_min=self.load_lower_limit,
                check_voltage=self.check_voltage,
                check_slack=self.check_slack,
            )
            if self.check_dynamic_constraints() > 0:
                break
        obs, _ = self._get_observation()
        return obs

    # get observation
    def _get_observation(self):
        api = self.psops
        converge = api.cal_power_flow_basic_nr()
        if converge > 0:
            bus_result = api.get_bus_all_lf_result()[:, 0]
            gen_result = api.get_generator_all_lf_result().reshape(
                api.get_generator_number() * 2, order="F"
            )
            load_result = api.get_load_all_lf_result().reshape(
                api.get_load_number() * 2, order="F"
            )
            section_p = self.cal_sections_power()
            self.state = np.concatenate([bus_result, gen_result, load_result])
            # self.state = np.concatenate([bus_result, gen_result, load_result, section_p])
            obs = self.state[self.ob_no]
        else:
            v = np.zeros(api.get_bus_number())
            gen_p = api.get_generator_all_p_set()
            gen_p[api.get_generator_all_slack()] = 0.0
            gen_q = np.zeros(api.get_generator_number())
            load_p = api.get_load_all_p_set()
            load_q = api.get_load_all_q_set()
            section_p = np.zeros(len(self.sections))
            self.state = np.concatenate([v, gen_p, gen_q, load_p, load_q])
            # self.state = np.concatenate([v, gen_p, gen_q, load_p, load_q, section_p])
            obs = self.state[self.ob_no]
            # obs = np.zeros(api.get_bus_number()+api.get_generator_number()*2+api.get_load_number()*2, dtype=np.float32)
        # obs = (obs - self.__centralOb) / self.__deltaOb
        return obs, converge
 
    # def get_random_action(self):
    #     api = self.psops
    #     api.get_power_flow_sample_simple_random(load_max=-1, load_min=-1)

    #     return self.cal_sections_power()/self.sections_pmax

    def step(self, cur_action, adjust_mode='ratio',goal='cost'):
        # act = cur_action * self.__deltaCtrl + self.__curCtrl
        # p_sections=cur_action
        p_sections=cur_action*self.sections_pmax
        if adjust_mode=='ratio':
            self.adjust_areas(p_sections)
            if self.check_static_constraints()>0:
                self.adjust_sensitivity() 
            obs, converge = self._get_observation()   
            rew = self._get_reward(obs, converge)
        elif adjust_mode=='sensitivity':
            success=self.adjust_sections_power_sensitivity(p_sections,goal)
            it=0
            while self.check_static_constraints()>0 and it<5:
                it+=1
                self.adjust_sensitivity()                
            obs, converge = self._get_observation()   
            if not success:
                rew=-1000
            else:
                rew = self._get_reward(obs, converge)
        else:
            obs, converge = self._get_observation()   
            rew = self._get_reward(obs, converge)   
            print('调整方式选择错误') 

        # don = not converge
        don = True
        inf = None
        return obs, rew, don, inf

    def cal_dynamic_criterion(self):
        api = self.psops
        stability_result = api.get_acsystem_all_ts_result()[0, :, 1]
        flg = False
        if np.any(stability_result == 0.0):  # early termination
            stop_step = np.where(stability_result == 0)[0]
            if stop_step[0] < 50:
                flg = True
        if flg == True:
            if self.reward_type == "unstabletime":
                criterion_value = (
                    api.get_info_ts_max_step() - 1
                ) * api.get_info_ts_delta_t()
            elif self.reward_type == "maxrotordiff":
                criterion_value = 500.0
            elif self.reward_type == "tsi":
                criterion_value = 1.0
        else:
            criterion_value = 0
            if self.reward_type == "unstabletime":
                stability_result = np.where(stability_result > self.criterion)[0]
                if stability_result.shape[0] != 0:
                    criterion_value += (
                        (api.get_info_ts_max_step() - 1) - stability_result[0]
                    ) * api.get_info_ts_delta_t()
            elif self.reward_type == "maxrotordiff":
                delta_max = min(self.criterion + 500.0, abs(stability_result).max())
                if delta_max > 0:
                    criterion_value = delta_max - self.criterion
            elif self.reward_type == "tsi":
                delta_max = min(999999.9, abs(stability_result).max())
                stability_result = (self.criterion - delta_max) / (
                    self.criterion + delta_max
                )
                if stability_result < 0:
                    criterion_value = 0 - stability_result
        return criterion_value

    def simulate_acline_fault(self, fault):
        api = self.psops
        acline_no = int(fault[0])
        terminal = int(fault[1])
        f_time = float(fault[2])
        assert (
            acline_no < api.get_acline_number()
            and terminal in [0, 100]
            and f_time >= 0.0
        ), f"acline fault set error, acline no {acline_no}, terminal {terminal}, f_time {f_time}"
        api.set_fault_disturbance_clear_all()
        api.set_fault_disturbance_add_acline(0, terminal, 0.0, f_time, acline_no)
        api.set_fault_disturbance_add_acline(1, terminal, f_time, 10.0, acline_no)
        api.cal_transient_stability_simulation_ti_sv()

    def check_dynamic_constraints(self):
        criterion_value = 0.0
        # check stability of anticipated fault set
        for fault in self.anticipated_fault_set:
            self.simulate_acline_fault(fault=fault)
            criterion_value += self.cal_dynamic_criterion()
        # check stability of anticipated disturbance set
        for disturbance in self.anticipated_disturbance_set:
            self.simulate_disturbance(disturbance=disturbance)
            criterion_value += self.cal_dynamic_criterion()
        return criterion_value
    
    def check_static_constraints(self, obs=None):
        api=self.psops
        if obs is None: obs, _ = self._get_observation()
        lower = self.state_space.low
        upper = self.state_space.high
        if self.static_check == 'all':
            limit = sum((self.state - upper)[self.state > upper]) + sum((lower - self.state)[self.state < lower])
        elif self.static_check == 'important':
            idx = self.important_state_idx
            limit = sum((self.state[idx] - upper[idx])[self.state[idx] > upper[idx]]) + sum((lower[idx] - self.state[idx])[self.state[idx] < lower[idx]])
        elif self.static_check == 'none':
            limit = 0.0
        else:
            limit = sum((self.state - upper)[self.state > upper]) + sum((lower - self.state)[self.state < lower])
        return limit

    def check_static_violation_no(self,obs=None):
        api=self.psops
        if obs is None: obs, _ = self._get_observation()
        lower = self.state_space.low
        upper = self.state_space.high
        out=np.array(self.state>upper) | np.array(self.state<lower)
        bus_number=api.get_bus_number()
        gen_number=api.get_generator_number()
        load_number=api.get_load_number()
        v_out_no=np.where(out[:bus_number])
        genp_out_no=np.where(out[bus_number:bus_number+gen_number])
        genq_out_no=np.where(out[bus_number+gen_number:bus_number+gen_number*2])
        load_out_no=np.where(out[-load_number*2:])
        out_no={'bus v out':np.array(v_out_no).reshape(-1),'generator q out': np.array(genq_out_no).reshape(-1),'generator p out:':np.array(genp_out_no).reshape(-1)}
        return out_no
     
    def _get_reward(self, obs, converge):
        api = self.psops
        if converge < 0: # not converge
            re = self.neg_max
        else:
            # check stability
            finish_time = self.check_dynamic_constraints()
            if finish_time > 0: # dynamic constraint violation
                re = max(self.neg_mid+1-self.c_fault*finish_time, self.neg_max+1)
                # print(self.c_fault, finish_time, re)
            else:
                limit = self.check_static_constraints(obs=obs)
                if limit > 0: # static constraint violation
                    re = max(self.co_static_violation*limit, self.neg_mid+1)
                else: # secure state
                    g = self.state[api.get_bus_number():api.get_bus_number()+api.get_generator_number()]
                    # cost = sum(0.2 + 30. * g + 100. * g * g)
                    cost = sum(self.ck1 + self.ck2 * g + self.ck3 * g * g)
                    re = max(self.co_reward * (1 - cost / self.max_cost), 0)
                    # re = max(1000. - 0.01 * cost, 0.0)
                    # re = max(3200 - 0.0001 * cost, 0.0)
        return re

    def get_max_cost(self):
        return self.__max_cost

    def cal_action(self, x):
        if x.ndim == 1:
            _, y, _, _ = self.step(x)
        elif x.ndim == 2:
            y = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                _, y[i], _, _ = self.step(x[i])
        else:
            raise Exception("wrong dimension")
        return -y


@ray.remote
class worker_sections(sections_Env):
    def __init__(
        self,
        flg=0,
        rng=None,
        sys_no=-1,
        act_gen_v=None,
        act_gen_p=None,
        sampler="stepwise",
        static_check="all",
        observation_type="minimum",  # minimum state, all state
        action_type="absolute",  # absolute action, or delta action
        reward_type="unstabletime",  # maximum rotor angle, TSI, unstable duration
        check_voltage=True,
        check_slack=True,
        upper_load=1.2,
        lower_load=0.7,
        criterion=180.0,
        fault_set=[[26, 0, 0.1], [26, 100, 0.1]],
        disturbance_set=[],
        co_reward=1000.0,
    ):
        super().__init__(
            flg=flg,
            rng=rng,
            sys_no=sys_no,
            act_gen_v=act_gen_v,
            act_gen_p=act_gen_p,
            sampler=sampler,
            static_check=static_check,
            check_voltage=check_voltage,
            observation_type=observation_type,
            action_type=action_type,
            reward_type=reward_type,
            check_slack=check_slack,
            upper_load=upper_load,
            lower_load=lower_load,
            criterion=criterion,
            fault_set=fault_set,
            disturbance_set=disturbance_set,
            co_reward=co_reward,
        )
        self.__worker_no = flg

    def get_work_no(self):
        return self.__worker_no


from tqdm import tqdm
if __name__ == "__main__":
    import time
    # import gym
    t1 = time.time()
    rng = np.random.default_rng(1024)
    env = sections_Env(rng=rng)
    api=env.psops
    n=100
    for i in tqdm(range(n)):
        while 1:
            env.set_random_start()
            limit=env.check_static_constraints()
            if limit>0:
                break
        env.step(0.2)
        obs,converge=env._get_observation()
        reward=env._get_reward(obs,converge)
        violation_no=env.check_static_violation_no(obs)
        print(reward)
        print(violation_no)