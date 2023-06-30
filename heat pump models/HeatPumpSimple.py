"""Heat pump model with basic cycle."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI as PSI
from fluprodia import FluidPropertyDiagram
from scipy.interpolate import interpn
from sklearn.linear_model import LinearRegression
from tespy.components import (Compressor, Condenser, CycleCloser,
                              DropletSeparator, HeatExchanger,
                              HeatExchangerSimple, Merge, Pump, Sink, Source,
                              Splitter, Valve)
from tespy.connections import Bus, Connection, Ref
from tespy.networks import Network
from tespy.tools import ExergyAnalysis
from tespy.tools.characteristics import CharLine
from tespy.tools.characteristics import load_default_char as ldc


class HeatPumpSimple():
    """Heat pump cycle with economizer."""

    def __init__(self, params):
        """Initialize model and set necessary attributes."""
        self.params = params

        self.wf = self.params['fluids']['wf']
        self.si = self.params['fluids']['si']
        self.so = self.params['fluids']['so']

        if self.si == self.so:
            self.fluid_vec_wf = {self.wf: 1, self.si: 0}
            self.fluid_vec_si = {self.wf: 0, self.si: 1}
            self.fluid_vec_so = {self.wf: 0, self.si: 1}
        else:
            self.fluid_vec_wf = {self.wf: 1, self.si: 0, self.so: 0}
            self.fluid_vec_si = {self.wf: 0, self.si: 1, self.so: 0}
            self.fluid_vec_so = {self.wf: 0, self.si: 0, self.so: 1}

        self.comps = dict()
        self.conns = dict()
        self.busses = dict()

        self.nw = Network(
            fluids=[fluid for fluid in self.fluid_vec_wf],
            T_unit='C', p_unit='bar', h_unit='kJ / kg',
            m_unit='kg / s'
            )

        self.cop = np.nan
        self.epsilon = np.nan
        self.solved_design = False

    def generate_components(self):
        """Initialize components of heat pump."""
        # Heat source
        self.comps['hs_ff'] = Source('Heat Source Feed Flow')
        self.comps['hs_bf'] = Sink('Heat Source Back Flow')
        self.comps['hs_pump'] = Pump('Heat Source Recirculation Pump')

        # Heat sink
        self.comps['cons_cc'] = CycleCloser('Consumer Cycle Closer')
        self.comps['cons_pump'] = Pump('Consumer Recirculation Pump')
        self.comps['cons'] = HeatExchangerSimple('Consumer')

        # Main cycle
        self.comps['cond'] = Condenser('Condenser')
        self.comps['cc'] = CycleCloser('Main Cycle Closer')
        self.comps['valve'] = Valve('EValve')
        self.comps['evap'] = HeatExchanger('Evaporator')
        self.comps['comp'] = Compressor('Compressor')

    def generate_connections(self):
        """Initialize and add connections and busses to network."""
        # Connections
        self.conns['A0'] = Connection(
            self.comps['cond'], 'out1', self.comps['cc'], 'in1', 'A0'
            )
        self.conns['A1'] = Connection(
            self.comps['cc'], 'out1', self.comps['valve'], 'in1', 'A1'
            )
        self.conns['A2'] = Connection(
            self.comps['valve'], 'out1', self.comps['evap'], 'in2', 'A2'
            )
        self.conns['A3'] = Connection(
            self.comps['evap'], 'out2', self.comps['comp'], 'in1', 'A3'
            )
        self.conns['A4'] = Connection(
            self.comps['comp'], 'out1', self.comps['cond'], 'in1', 'A4'
            )

        self.conns['B1'] = Connection(
            self.comps['hs_ff'], 'out1', self.comps['evap'], 'in1', 'B1'
            )
        self.conns['B2'] = Connection(
            self.comps['evap'], 'out1', self.comps['hs_pump'], 'in1', 'B2'
            )
        self.conns['B3'] = Connection(
            self.comps['hs_pump'], 'out1', self.comps['hs_bf'], 'in1', 'B3'
            )

        self.conns['C0'] = Connection(
            self.comps['cons'], 'out1', self.comps['cons_cc'], 'in1', 'C0'
            )
        self.conns['C1'] = Connection(
            self.comps['cons_cc'], 'out1', self.comps['cons_pump'], 'in1', 'C1'
            )
        self.conns['C2'] = Connection(
            self.comps['cons_pump'], 'out1', self.comps['cond'], 'in2', 'C2'
            )
        self.conns['C3'] = Connection(
            self.comps['cond'], 'out2', self.comps['cons'], 'in1', 'C3'
            )

        self.nw.add_conns(*[conn for conn in self.conns.values()])

        # Busses
        mot_x = np.array([
            0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
            0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15,
            1.2, 10
            ])
        mot_y = (np.array([
            0.01, 0.3148, 0.5346, 0.6843, 0.7835, 0.8477, 0.8885, 0.9145,
            0.9318, 0.9443, 0.9546, 0.9638, 0.9724, 0.9806, 0.9878, 0.9938,
            0.9982, 1.0009, 1.002, 1.0015, 1, 0.9977, 0.9947, 0.9909, 0.9853,
            0.9644
            ]) * 0.98)
        mot = CharLine(x=mot_x, y=mot_y)
        self.busses['power input'] = Bus('power input')
        self.busses['power input'].add_comps(
            {'comp': self.comps['comp'], 'char': mot, 'base': 'bus'},
            {'comp': self.comps['hs_pump'], 'char': mot, 'base': 'bus'},
            {'comp': self.comps['cons_pump'], 'char': mot, 'base': 'bus'}
            )

        self.busses['heat input'] = Bus('heat input')
        self.busses['heat input'].add_comps(
            {'comp': self.comps['hs_ff'], 'base': 'bus'},
            {'comp': self.comps['hs_bf'], 'base': 'component'}
            )

        self.busses['heat output'] = Bus('heat output')
        self.busses['heat output'].add_comps(
            {'comp': self.comps['cons'], 'base': 'component'}
            )

        self.nw.add_busses(*[bus for bus in self.busses.values()])

    def init_simulation(self, **kwargs):
        """Perform initial parametrization with starting values."""
        # Components
        self.comps['comp'].set_attr(eta_s=self.params['comp']['eta_s'])
        self.comps['hs_pump'].set_attr(eta_s=self.params['hs_pump']['eta_s'])
        self.comps['cons_pump'].set_attr(
            eta_s=self.params['cons_pump']['eta_s']
            )

        self.comps['evap'].set_attr(
            pr1=self.params['evap']['pr1'], pr2=self.params['evap']['pr2']
            )
        self.comps['cond'].set_attr(
            pr1=self.params['cond']['pr1'], pr2=self.params['cond']['pr2']
            )
        self.comps['cons'].set_attr(
            pr=self.params['cons']['pr'], Q=self.params['cons']['Q']
            , dissipative=False
            )

        # Connections
        # Starting values
        p_evap = PSI(
            'P', 'Q', 1,
            'T', self.params['B2']['T'] - self.params['evap']['ttd_l'] + 273.15,
            self.wf
            ) * 1e-5
        p_cond = PSI(
            'P', 'Q', 0,
            'T', self.params['C3']['T'] + self.params['cond']['ttd_u'] + 273.15,
            self.wf
            ) * 1e-5

        # Main cycle
        self.conns['A3'].set_attr(x=self.params['A3']['x'], p=p_evap)
        self.conns['A0'].set_attr(p=p_cond, fluid=self.fluid_vec_wf)
        # Heat source
        self.conns['B1'].set_attr(
            T=self.params['B1']['T'], p=self.params['B1']['p'],
            fluid=self.fluid_vec_so
            )
        self.conns['B2'].set_attr(T=self.params['B2']['T'])
        self.conns['B3'].set_attr(p=self.params['B1']['p'])

        # Heat sink
        self.conns['C3'].set_attr(
            T=self.params['C3']['T'], p=self.params['C3']['p'],
            fluid=self.fluid_vec_si
            )
        self.conns['C0'].set_attr(T=self.params['C0']['T'])

        # Perform initial simulation and unset starting values
        self._solve_model(**kwargs)

        self.conns['A3'].set_attr(p=None)
        self.conns['A0'].set_attr(p=None)

    def design_simulation(self, **kwargs):
        """Perform final parametrization and design simulation."""
        self.comps['evap'].set_attr(ttd_l=self.params['evap']['ttd_l'])
        self.comps['cond'].set_attr(ttd_u=self.params['cond']['ttd_u'])

        self._solve_model(**kwargs)

        if self.nw.res[-1] <= 1e-3:
            self.solved_design = True
            self.design_path = 'hp_simple_design'
            self.nw.save(self.design_path)

        self.m_design = self.conns['A4'].m.val

        self.cop = (
            abs(self.busses['heat output'].P.val)
            / self.busses['power input'].P.val
            )

    def _solve_model(self, **kwargs):
        """Solve the model in design mode."""
        if 'iterinfo' in kwargs:
            self.nw.set_attr(iterinfo=kwargs['iterinfo'])
        self.nw.solve('design')
        if 'print_results' in kwargs:
            if kwargs['print_results']:
                self.nw.print_results()

    def run_model(self, **kwargs):
        """Run the initialization and design simulation routine."""
        self.generate_components()
        self.generate_connections()
        self.init_simulation(**kwargs)
        self.design_simulation(**kwargs)

    def offdesign_simulation(self):
        """Perform offdesign parametrization and simulation."""
        if not self.solved_design:
            raise RuntimeError(
                'Heat pump has not been designed via the "design_simulation" '
                + 'method. Therefore the offdesign simulation will fail.'
                )

        # Parametrization
        self.comps['comp'].set_attr(
            design=['eta_s'], offdesign=['eta_s_char']
            )
        self.comps['hs_pump'].set_attr(
            design=['eta_s'], offdesign=['eta_s_char']
            )
        self.comps['cons_pump'].set_attr(
            design=['eta_s'], offdesign=['eta_s_char']
            )

        self.conns['B1'].set_attr(offdesign=['v'])
        self.conns['B2'].set_attr(design=['T'])

        self.comps['cond'].set_attr(
            design=['pr2', 'ttd_u'], offdesign=['zeta2', 'kA_char']
            )
        self.comps['cons'].set_attr(design=['pr'], offdesign=['zeta'])

        kA_char1 = ldc('heat exchanger', 'kA_char1', 'DEFAULT', CharLine)
        kA_char2 = ldc(
            'heat exchanger', 'kA_char2', 'EVAPORATING FLUID', CharLine
            )
        self.comps['evap'].set_attr(
            kA_char1=kA_char1, kA_char2=kA_char2,
            design=['pr1', 'ttd_l'], offdesign=['zeta1', 'kA_char']
            )

        # Simulation
        self.T_hs_ff_range = np.linspace(
            self.params['offdesign']['T_hs_ff_start'],
            self.params['offdesign']['T_hs_ff_end'],
            self.params['offdesign']['T_hs_ff_steps'],
            endpoint=True
            )
        self.T_cons_ff_range = np.linspace(
            self.params['offdesign']['T_cons_ff_start'],
            self.params['offdesign']['T_cons_ff_end'],
            self.params['offdesign']['T_cons_ff_steps'],
            endpoint=True
            )
        self.pl_range = np.linspace(
            self.params['offdesign']['partload_min'],
            self.params['offdesign']['partload_max'],
            self.params['offdesign']['partload_steps'],
            endpoint=True
            )
        deltaT_hs = (
            self.params['B1']['T']
            - self.params['B2']['T']
            )

        self.Q_array = list()
        self.P_array = list()
        self.offdesign_results = dict()

        for T_hs_ff in self.T_hs_ff_range:
            Q_subarray = list()
            P_subarray = list()
            self.offdesign_results[T_hs_ff] = dict()
            self.conns['B1'].set_attr(T=T_hs_ff)
            self.conns['B2'].set_attr(T=T_hs_ff-deltaT_hs)
            for T_cons_ff in self.T_cons_ff_range:
                self.conns['C3'].set_attr(T=T_cons_ff)
                Q_subsubarray = list()
                P_subsubarray = list()
                self.offdesign_results[T_hs_ff][T_cons_ff] = dict()
                for pl in self.pl_range[::-1]:
                    print(
                        f'### Temp. HS = {T_hs_ff} °C, Temp. Cons = '
                        + f'{T_cons_ff} °C, Partload = {pl*100} % ###'
                        )
                    self.init_path = None
                    no_init_path = (
                        (T_cons_ff != self.T_cons_ff_range[0])
                        and (pl == self.pl_range[-1])
                        )
                    if no_init_path:
                        self.init_path = 'hp_simple_init'
                    self.comps['cons'].set_attr(Q=None)
                    self.conns['A1'].set_attr(m=pl*self.m_design)

                    self.nw.solve(
                        'offdesign', design_path=self.design_path,
                        init_path=self.init_path
                        )

                    if pl == self.pl_range[-1] and self.nw.res[-1] < 1e-3:
                        self.nw.save('hp_simple_init')

                    Q_subsubarray += [self.busses['heat output'].P.val * 1e-6]
                    P_subsubarray += [self.busses['power input'].P.val * 1e-6]
                    self.offdesign_results[T_hs_ff][T_cons_ff][pl] = {
                        'P': P_subsubarray[-1],
                        'Q': abs(Q_subsubarray[-1]),
                        'COP':  abs(Q_subsubarray[-1])/P_subsubarray[-1]
                        }
                Q_subarray += [Q_subsubarray[::-1]]
                P_subarray += [P_subsubarray[::-1]]
            self.Q_array += [Q_subarray]
            self.P_array += [P_subarray]
        if self.params['offdesign']['save_results']:
            with open('hp_simple_partload.json', 'w') as file:
                json.dump(self.offdesign_results, file, indent=4)

    def calc_partload_char(self, **kwargs):
        """
        Interpolate data points of heat output and power input.

        Return functions to interpolate values heat output and
        power input based on the partload and the feed flow
        temperatures of the heat source and sink. If there is
        no data given through keyword arguments, the instances
        attributes will be searched for the necessary data.

        Parameters
        ----------
        kwargs : dict
            Necessary data is:
                Q_array : 3d array
                P_array : 3d array
                pl_rage : 1d array
                T_hs_ff_range : 1d array
                T_cons_ff_range : 1d array
        """
        necessary_params = [
            'Q_array', 'P_array', 'pl_range', 'T_hs_ff_range',
            'T_cons_ff_range'
            ]
        if len(kwargs):
            for nec_param in necessary_params:
                if nec_param not in kwargs:
                    raise KeyError(
                        f'Necessary parameter {nec_param} not '
                        + 'in kwargs. The necessary parameters'
                        + f' are: {necessary_params}'
                        )
            Q_array = np.asarray(kwargs['Q_array'])
            P_array = np.asarray(kwargs['P_array'])
            pl_range = kwargs['pl_range']
            T_hs_ff_range = kwargs['T_hs_ff_range']
            T_cons_ff_range = kwargs['T_cons_ff_range']
        else:
            for nec_param in necessary_params:
                if nec_param not in self.__dict__:
                    raise AttributeError(
                        f'Necessary parameter {nec_param} can '
                        + 'not be found in the instances '
                        + 'attributes. Please make sure to '
                        + 'perform the offdesign_simulation '
                        + 'method or provide the necessary '
                        + 'parameters as kwargs. These are: '
                        + f'{necessary_params}'
                        )
            Q_array = np.asarray(self.Q_array)
            P_array = np.asarray(self.P_array)
            pl_range = self.pl_range
            T_hs_ff_range = self.T_hs_ff_range
            T_cons_ff_range = self.T_cons_ff_range

        pl_step = 0.01
        T_hs_ff_step = 1
        T_cons_ff_step = 1

        pl_fullrange = np.arange(
            pl_range[0],
            pl_range[-1]+pl_step,
            pl_step
            )
        T_hs_ff_fullrange = np.arange(
            T_hs_ff_range[0], T_hs_ff_range[-1]+T_hs_ff_step, T_hs_ff_step
            )
        T_cons_ff_fullrange = np.arange(
            T_cons_ff_range[0], T_cons_ff_range[-1]+T_cons_ff_step,
            T_cons_ff_step
            )

        multiindex = pd.MultiIndex.from_product(
                        [T_hs_ff_fullrange, T_cons_ff_fullrange, pl_fullrange],
                        names=['T_hs_ff', 'T_cons_ff', 'pl']
                        )

        partload_char = pd.DataFrame(
            index=multiindex, columns=['Q', 'P', 'COP']
            )

        for T_hs_ff in T_hs_ff_fullrange:
            for T_cons_ff in T_cons_ff_fullrange:
                for pl in pl_fullrange:
                    partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'Q'] = abs(
                        interpn(
                            (T_hs_ff_range, T_cons_ff_range, pl_range),
                            Q_array,
                            (round(T_hs_ff, 3), round(T_cons_ff, 3),
                             round(pl, 3)),
                            bounds_error=False
                            )[0]
                        )
                    partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'P'] = interpn(
                        (T_hs_ff_range, T_cons_ff_range, pl_range),
                        P_array,
                        (round(T_hs_ff, 3), round(T_cons_ff, 3), round(pl, 3)),
                        bounds_error=False
                        )[0]
                    partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'COP'] = (
                        partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'Q']
                        / partload_char.loc[(T_hs_ff, T_cons_ff, pl), 'P']
                        )

        return partload_char

    def linearize_partload_char(self, partload_char, line_type='offset',
                                regression_type='OLS'):
        """
        Linearize partload characteristic for usage in MILP problems.

        Parameters
        ----------
        partload_char : pd.DataFrame
            DataFrame of the full partload characteristic containing 'Q', 'P'
            and 'COP' with a MultiIndex of the three variables 'T_hs_ff',
            'T_cons_ff' and 'pl'.

        line_type : str
            Type of linear model to generate. Options are 'origin' for a line
            through the origin or 'offset' for mixed integer offset model.
            Defaults to 'offset' if it is not set.

        regression_type : str
            Type of regression method to use for linearization of the partload
            characteristic. Options are 'OLS' for the method of ordinary least
            squares or 'MinMax' for a line from the minimum to the maximum
            value.
            Defaults to 'OLS' if it is not set.
        """
        cols = ['P_max', 'P_min']
        if line_type == 'origin':
            cols += ['COP']
        elif line_type == 'offset':
            cols += ['c_1', 'c_0']

        T_hs_ff_range = set(
            partload_char.index.get_level_values('T_hs_ff')
            )
        T_cons_ff_range = set(
            partload_char.index.get_level_values('T_cons_ff')
            )

        multiindex = pd.MultiIndex.from_product(
            [T_hs_ff_range, T_cons_ff_range],
            names=['T_hs_ff', 'T_cons_ff']
            )
        linear_model = pd.DataFrame(index=multiindex, columns=cols)

        for T_hs_ff in T_hs_ff_range:
            for T_cons_ff in T_cons_ff_range:
                idx = (T_hs_ff, T_cons_ff)
                linear_model.loc[idx, 'P_max'] = (
                    partload_char.loc[idx, 'P'].max()
                    )
                linear_model.loc[idx, 'P_min'] = (
                    partload_char.loc[idx, 'P'].min()
                    )
                if regression_type == 'MinMax':
                    if line_type == 'origin':
                        linear_model.loc[idx, 'COP'] = (
                            partload_char.loc[idx, 'Q'].max()
                            / partload_char.loc[idx, 'P'].max()
                            )
                    elif line_type == 'offset':
                        linear_model.loc[idx, 'c_1'] = (
                            (partload_char.loc[idx, 'Q'].max()
                             - partload_char.loc[idx, 'Q'].min())
                            / (partload_char.loc[idx, 'P'].max()
                               - partload_char.loc[idx, 'P'].min())
                            )
                        linear_model.loc[idx, 'c_0'] = (
                            partload_char.loc[idx, 'Q'].max()
                            - partload_char.loc[idx, 'P'].max()
                            * linear_model.loc[idx, 'c_1']
                            )
                elif regression_type == 'OLS':
                    regressor = partload_char.loc[idx, 'P'].to_numpy()
                    regressor = regressor.reshape(-1, 1)
                    response = partload_char.loc[idx, 'Q'].to_numpy()
                    if line_type == 'origin':
                        LinReg = LinearRegression(fit_intercept=False).fit(
                            regressor, response
                            )
                        linear_model.loc[idx, 'COP'] = LinReg.coef_[0]
                    elif line_type == 'offset':
                        LinReg = LinearRegression().fit(regressor, response)
                        linear_model.loc[idx, 'c_1'] = LinReg.coef_[0]
                        linear_model.loc[idx, 'c_0'] = LinReg.intercept_

        return linear_model

    def arrange_char_timeseries(self, linear_model, temp_ts):
        """
        Arrange a timeseries of the characteristics based on temperature data.

        If T_cons_ff in temperature timeseries is out of bounds, the closest
        characteristic (min. or max. temperature).

        Parameters
        ----------
        linear_model : pd.DataFrame
            DataFrame of the linearized partload characteristic with a
            MultiIndex of the three variables 'T_hs_ff' and 'T_cons_ff'.

        temp_ts : pd.DataFrame
            Timeseries of 'T_hs_ff' and 'T_cons_ff' as they occur in the period
            observed.
        """
        char_ts = pd.DataFrame(
            index=temp_ts.index, columns=linear_model.columns
            )
        for i in temp_ts.index:
            try:
                char_ts.loc[i, :] = linear_model.loc[
                    (temp_ts.loc[i, 'T_hs_ff'], temp_ts.loc[i, 'T_cons_ff']), :
                    ]
            except KeyError:
                print(temp_ts.loc[i, 'T_cons_ff'], 'not in linear_model.')
                T_cons_ff_range = linear_model.index.get_level_values('T_cons_ff')
                if temp_ts.loc[i, 'T_cons_ff'] < min(T_cons_ff_range):
                    multi_idx = (
                        temp_ts.loc[i, 'T_hs_ff'], min(T_cons_ff_range)
                        )
                elif temp_ts.loc[i, 'T_cons_ff'] > max(T_cons_ff_range):
                    multi_idx = (
                        temp_ts.loc[i, 'T_hs_ff'], max(T_cons_ff_range)
                        )
                char_ts.loc[i, :] = linear_model.loc[multi_idx, :]

        return char_ts

    def get_log_ph_states(self):
        """Get plotting data for components to be shown in log(p)-h-diagram."""
        data = {}
        data.update(
            {self.comps['cond'].label:
             self.comps['cond'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['valve'].label:
             self.comps['valve'].get_plotting_data()[1]}
        )
        data.update(
            {self.comps['evap'].label:
             self.comps['evap'].get_plotting_data()[2]}
        )
        data.update(
            {self.comps['comp'].label:
             self.comps['comp'].get_plotting_data()[1]}
        )

        return data

    def plot_logph(self, savefig=False, return_diagram=False):
        """Generate log(p)-h-diagram of heat pump process."""
        result_dict = self.get_log_ph_states()

        # Initialize fluid property diagram
        diagram = FluidPropertyDiagram(self.wf)
        diagram.set_unit_system(T='°C', p='bar', h='kJ/kg')
        diagram.set_limits(
            x_min=self.params['logph']['x_min'],
            x_max=self.params['logph']['x_max'],
            y_min=self.params['logph']['y_min'],
            y_max=self.params['logph']['y_max']
            )

        # Calculate components process data
        for compdata in result_dict.values():
            compdata['datapoints'] = (
                diagram.calc_individual_isoline(**compdata)
                )

        # Isolines
        diagram.calc_isolines()
        diagram.draw_isolines('logph')

        # Draw heat pump process over fluid property diagram
        for comp, compdata in result_dict.items():
            datapoints = compdata['datapoints']
            diagram.ax.plot(datapoints['h'], datapoints['p'], color='#EC6707')
            diagram.ax.scatter(
                datapoints['h'][0], datapoints['p'][0],  #  color='#B54036',
                label=comp
                )

        diagram.ax.legend()

        if savefig:
            filepath = (
                f'logph_{self.params["setup"]["type"]}_'
                + f'{self.params["setup"]["refrig"]}.pdf'
                )
            diagram.save(filepath, dpi=300)
            os.startfile(filepath)
        else:
            plt.show()

        if return_diagram:
            return diagram

    def perform_exergy_analysis(self, print_results=False, **kwargs):
        """Perform exergy analysis."""
        self.ean = ExergyAnalysis(
            self.nw,
            E_F=[self.busses['power input'], self.busses['heat input']],
            E_P=[self.busses['heat output']]
            )
        self.ean.analyse(
            pamb=self.params['ambient']['p'], Tamb=self.params['ambient']['T']
            )
        if print_results:
            self.ean.print_results(**kwargs)

        self.epsilon = self.ean.network_data['epsilon']


if __name__ == '__main__':
    import json
    import os

    with open('params_hp_simple_4GDH.json', 'r') as file:
        params = json.load(file)

    hp = HeatPumpSimple(params)

    hp.run_model()
    # hp.plot_logph(savefig=True)
    hp.offdesign_simulation()
    partload_char = hp.calc_partload_char()
    linear_model = hp.linearize_partload_char(partload_char)

    # datapath = os.path.join(
    #     os.path.abspath(__file__),
    #     '..', '..', '..',
    #     '06 - Wärmenetze', 'Einsatzoptimierung', 'primary_network', 'input',
    #     'pn16hpsimple_data.csv'
    # )

    # datapath = os.path.join(
    #     os.path.abspath(__file__),
    #     '..', '..', '..',
    #     '06 - Wärmenetze', 'Einsatzoptimierung', 'sub_network', 'input',
    #     'sn16_data.csv'
    # )

    # data = pd.read_csv(datapath, sep=';', index_col=0, parse_dates=True)

    # temppath = os.path.join(
    #     os.path.abspath(__file__),
    #     '..', '..', '..',
    #     '06 - Wärmenetze', 'Einsatzoptimierung', 'misc',
    #     'TempTimeseries2016.csv'
    # )

    temppath = os.path.join(
        os.path.abspath(__file__),
        '..', '..', '..',
        'sfjv_dhs_data_2019_hourly_noDST_rounded.csv'
    )

    temp_ts = pd.read_csv(temppath, sep=';', index_col=0, parse_dates=True)
    temp_ts['T_hs_ff'] = 10
    temp_ts = temp_ts[['plant5_temp_feed_flow', 'T_hs_ff']]
    # temp_ts.rename(columns={'T_VL': 'T_cons_ff'}, inplace=True)

    temp_ts.rename(columns={'plant5_temp_feed_flow': 'T_cons_ff'}, inplace=True)

    temp_ts['T_cons_ff'] = temp_ts['T_cons_ff'].apply(int)

    data = pd.DataFrame()
    data['T_ff'] = temp_ts['T_cons_ff']

    char_ts = hp.arrange_char_timeseries(linear_model, temp_ts)
    for col in char_ts.columns:
        data[f'hp_{col}'] = char_ts[col]

    if not os.path.exists(params['fluids']['wf']):
        os.mkdir(params['fluids']['wf'])

    # data.to_csv(datapath, sep=';')
    data.to_csv(
        os.path.join(params['fluids']['wf'], 'hp_simple_4GDH.csv'), sep=';'
        )
