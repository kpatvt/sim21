"""Models Envelope

Classes:
PTEnvelope   -- Class for 2-phases vapour, liquid PT envelope
QualityCurve -- Class for a single curve in the envelope

"""
# import numpy as np
import re
from sim21.solver.Variables import *
from sim21.thermo.ThermoAdmin import EnvelopeResults
from sim21.unitop.Pump import DataSeries
from sim21.unitop import Stream
from sim21.kludges import cmp

THCURVE_INDICATOR = 'TH'
PHCURVE_INDICATOR = 'PH'
SATRUATED_CURVE = 'Saturation'

HYDRATE_PAR = 'Hydrate'
HYDRATESTATUS = 'HydrateStatus'
HYDRATERESULTS = 'HydrateResults'

DRYBASIS_PAR = 'DryBasis'  # whether to strip out water
NUPOINT_PAR = 'MaxNumPoints'  # max allowable number of points in each quality line
CRIT_P = 'Crit_P'
CRIT_T = 'Crit_T'
CRICONDENBAR_P = 'Cricondenbar_P'
CRICONDENBAR_T = 'Cricondenbar_T'
CRICONDENTHERM_P = 'Cricondentherm_P'
CRICONDENTHERM_T = 'Cricondentherm_T'
STARTING_P_PORT = 'Starting_P'
CURVERESULTS = 'Results'  # to access the QualityCurve EnvelopeResults
PRESSURESETPOINTS = 'Pressures'  # to access the Envelope pSets
ONDRYBASIS = 'OnDryBasis'  # whether the result is on dry basis or not
QUALITYCURVES = 'QualityCurves'
ENTHALPYCURVES = 'EnthalpyCurves'
INITP_PAR = 'InitPressures'


class QualityCurve(object):
    def __init__(self, vapFrac, initP=None):
        self.initP = initP
        self.vapFrac = vapFrac
        self.results = None
        self.parentEnvelope = None
        self.name = ''

    def __str__(self):
        s = re.sub(' .*', '', repr(self))[1:]
        if self.vapFrac:
            s += '; Identifier = %f' % self.vapFrac
        else:
            s += '; Identifier = '
        s += '; EnvelopeResults: ' + CURVERESULTS
        return s

    def CleanUp(self):
        self.parentEnvelope = None

    def Initialize(self, envelopeObj, name):
        self.parentEnvelope = envelopeObj
        self.name = name

    def GetPath(self):
        return self.parentEnvelope.GetPath() + '.' + self.name

    def GetParent(self):
        return self.parentEnvelope

    def SetResults(self, results=None):
        self.results = results

    def GetContents(self):
        result = [('Identifier', self.vapFrac),
                  (CURVERESULTS, self.results)]
        return result

    def GetObject(self, name):
        if name == CURVERESULTS:
            return self.results

    def SetParent(self, parent):
        """Used when cloning"""
        self.parentEnvelope = parent

    def Clone(self):
        clone = self.__class__(self.vapFrac, self.initP)
        clone.name = self.name
        if self.results:
            clone.results = copy.deepcopy(self.results)
        return clone


class PTEnvelope(Stream.Stream_Material):
    """Class for 2-phases VL PT envelope. Inherits from Stream_Material"""

    def __init__(self, initScript=None):
        """Init the PT Envelope"""
        Stream.Stream_Material.__init__(self, initScript)
        self.QualityLines = {}

        # Hydrate
        self.SetParameterValue(HYDRATE_PAR, 0)
        self.hydrateStatus = ''
        self.hydrateResults = None

        self.SetParameterValue(DRYBASIS_PAR, 1)
        self.SetParameterValue(NUPOINT_PAR, 50)
        self.pSet = DataSeries(P_VAR)
        self.pSet.Initialize(self, PRESSURESETPOINTS)
        self.pSet.SetType(P_VAR)
        self._strippedH2O = 0
        # add cricondenbar, cricondentherm, critical ports
        self.CreateSigPort(CRIT_P, P_VAR)
        self.CreateSigPort(CRIT_T, T_VAR)
        self.CreateSigPort(CRICONDENBAR_P, P_VAR)
        self.CreateSigPort(CRICONDENBAR_T, T_VAR)
        self.CreateSigPort(CRICONDENTHERM_P, P_VAR)
        self.CreateSigPort(CRICONDENTHERM_T, T_VAR)
        p = self.CreateSigPort(STARTING_P_PORT, P_VAR)
        p.SetValue(101.325, FIXED_V)

        # enthalpy curves
        self.thEnvelope = None
        self.phEnvelope = None

    def CleanUp(self):
        self.pSet.CleanUp()
        self.pSet = None
        for c in list(self.QualityLines.values()):
            c.CleanUp()
        self.QualityLines.clear()
        if self.thEnvelope:
            self.thEnvelope.CleanUp()
        if self.phEnvelope:
            self.phEnvelope.CleanUp()
        super(PTEnvelope, self).CleanUp()

    def CreateSigPort(self, name, varType):
        p = self.CreatePort(SIG, name)
        p.SetSignalType(varType)
        p.SetLocked(True)
        return p

    def GetContents(self):
        result = super(PTEnvelope, self).GetContents()
        result.append((ONDRYBASIS, self._strippedH2O))
        result.append((PRESSURESETPOINTS, self.pSet))
        # result.append(('NumQualityCurves', len(self.QualityLines)))
        sortedKeys = list(self.QualityLines.keys())[:]
        sortedKeys.sort(lambda a, b: cmp(self.QualityLines[a].vapFrac, self.QualityLines[b].vapFrac))
        result.append((QUALITYCURVES, sortedKeys))
        for key in list(self.QualityLines.keys()):
            result.append((self.QualityLines[key].name, self.QualityLines[key]))
        # hydrate
        hydrate = self.GetParameterValue(HYDRATE_PAR)
        if hydrate:
            result.append((HYDRATESTATUS, self.hydrateStatus))
            result.append((HYDRATERESULTS, self.hydrateResults))
        # TH Curves
        if self.thEnvelope:
            thData = self.thEnvelope.GetContents()
            result.extend(thData)
        if self.phEnvelope:
            phData = self.phEnvelope.GetContents()
            result.extend(phData)
        return result

    def GetObject(self, name):
        if name in self.QualityLines:
            return self.QualityLines[name]
        elif name == PRESSURESETPOINTS:
            return self.pSet
        elif name == HYDRATERESULTS:
            return self.hydrateResults
        if self.thEnvelope:
            if self.thEnvelope.name == name:
                return self.thEnvelope
            elif name in self.thEnvelope.QualityLines:
                return self.thEnvelope.QualityLines[name]
        if self.phEnvelope:
            if self.phEnvelope.name == name:
                return self.phEnvelope
            elif name in self.phEnvelope.QualityLines:
                return self.phEnvelope.QualityLines[name]
        return super(PTEnvelope, self).GetObject(name)

    def AddObject(self, obj, name):
        if isinstance(obj, QualityCurve):
            name = self.FreeCurveName(name)
            obj.Initialize(self, name)
            self.QualityLines[name] = obj
            for sigP in self.GetPorts(SIG):
                sigP.ForgetAllCalculations()
            self.PushSolveOp(self)
        elif isinstance(obj, PHEnvelope):
            self.phEnvelope = obj
            obj.Initialize(self, name)
            self.PushSolveOp(self)
        elif isinstance(obj, THEnvelope):
            self.thEnvelope = obj
            obj.Initialize(self, name)
            self.PushSolveOp(self)
        else:
            super(PTEnvelope, self).AddObject(obj, name)

    def DeleteObject(self, obj):
        if isinstance(obj, QualityCurve):
            name = obj.name
            if name in self.QualityLines:
                obj.CleanUp()
                del self.QualityLines[name]
                for sigP in self.GetPorts(SIG):
                    sigP.ForgetAllCalculations()
                self.PushSolveOp(self)
        elif isinstance(obj, THEnvelope):
            obj.CleanUp()
            if obj.name == self.phEnvelope.name:
                self.phEnvelope = None
            elif obj.name == self.thEnvelope.name:
                self.thEnvelope = None
        else:
            super(PTEnvelope, self).DeleteObject(obj)

    def SolvePorts(self):
        inPort = self.GetPort(IN_PORT)
        outPort = self.GetPort(OUT_PORT)

        while 1:
            # share between in and out
            inPort.ShareWith(outPort)

            # share with clones, if any
            for port in self.GetPorts(MAT | IN | OUT):
                inPort.ShareWith(port)

            inPort.ShareWith(outPort)

            for port in self.GetPorts(MAT | IN | OUT):
                outPort.ShareWith(port)

            # share between in and out again
            inPort.ShareWith(outPort)
            for port in self.GetPorts(MAT | IN | OUT):
                port.CalcFlows()

            # only have to check one flash since everything was shared
            if not self.Flash(outPort):
                break

    def Solve(self):
        # solve the stream
        self.SolvePorts()
        # reset envelop results
        criticalPoint = (None, None)
        cricondenbar = (None, None)
        cricondentherm = (None, None)
        self._strippedH2O = 0
        # self.hydrateResults = None
        # self.hydrateStatus = ''

        # check with Raul later, why do i need to solve during forget ?
        if self.IsForgetting():
            return

        # solve the envelope
        cmps = None
        self.unitOpMessage = ('NoMessage',)

        # calculate all required quality lines

        initPort = self.GetPort(STARTING_P_PORT)
        thCaseObj = self.GetThermo()

        '''
        # cannot do this, if i had a thermo case and then remove the case,
        # i need to remove the results before returning
        if thCaseObj == None:
            # quit if no thermoCase
            return
        '''
        if thCaseObj is None:
            cmps = None
        else:
            thAdmin, prov, case = thCaseObj.thermoAdmin, thCaseObj.provider, thCaseObj.case
            cmps = self.GetComposition(thAdmin, prov, case)

        # ensure the isobars are extrapolation points for the TH diagram
        pList = self.pSet.GetValues()
        if self.thEnvelope:
            for line in list(self.thEnvelope.QualityLines.values()):
                pSpec = line.vapFrac
                if pSpec > 0 and not (pSpec in pList):
                    pList.append(pSpec)

        for line in list(self.QualityLines.values()):
            if not cmps or thCaseObj is None:
                # i am not ready, remove the existing result (if any)
                line.SetResults(None)
            else:
                # use the global starting P, if not available, use the individual P
                initP = initPort.GetValue()
                if initP is None:
                    initP = line.initP
                    if initP is None:
                        initP = 101.325
                # since i may be completing points from both ends, double the points
                maxPoints = 2 * self.GetParameterValue(NUPOINT_PAR)
                # pList = self.pSet.GetValues()

                envResults = thAdmin.PhaseEnvelope(prov, case, cmps, line.vapFrac,
                                                   initP, maxPoints, pList)
                line.SetResults(envResults)
                # Set the error
                if envResults.pointCount == 0:
                    msg = line.name + ': ' + envResults.returnMessage
                    # envMsg += msg + "\n"
                    self.unitOpMessage = ('RawOutput', msg)

                # set the crit, cricondenbar, cricondentherm
                if line.vapFrac == 0.0 or line.vapFrac == 1.0:
                    # set crit, cricondenbar and crocondentherm from saturaion line
                    if envResults.criticalPoint[0] is not None:
                        criticalPoint = envResults.criticalPoint
                    if envResults.cricondenbar[0] is not None:
                        cricondenbar = envResults.cricondenbar
                    if envResults.cricondentherm[0] is not None:
                        cricondentherm = envResults.cricondentherm
                elif criticalPoint[0] is None and envResults.criticalPoint[0] is not None:
                    # critical point can be set from any quality curve
                    criticalPoint = envResults.criticalPoint
        # load the crit, cricondenbar, cricondentherm signals
        port = self.GetPort(CRIT_P)
        if port:
            port.SetValue(criticalPoint[0], CALCULATED_V)

        port = self.GetPort(CRIT_T)
        if port:
            port.SetValue(criticalPoint[1], CALCULATED_V)

        port = self.GetPort(CRICONDENBAR_P)
        if port:
            port.SetValue(cricondenbar[0], CALCULATED_V)

        port = self.GetPort(CRICONDENBAR_T)
        if port:
            port.SetValue(cricondenbar[1], CALCULATED_V)

        port = self.GetPort(CRICONDENTHERM_P)
        if port:
            port.SetValue(cricondentherm[0], CALCULATED_V)

        port = self.GetPort(CRICONDENTHERM_T)
        if port:
            port.SetValue(cricondentherm[1], CALCULATED_V)

        # hydrate curve
        hydrate = self.GetParameterValue(HYDRATE_PAR)
        if hydrate:
            try:
                if cricondenbar[0] is None:
                    if criticalPoint[0] is None:
                        pMax = 10000.0
                    else:
                        pMax = max(5000.0, criticalPoint[0])
                else:
                    pMax = max(5000.0, cricondenbar[0])
            except:
                pMax = 10000.0

            tMin = 260.0
            tMax = 310.0
            option = '%s %s %s' % (tMin, tMax, pMax)
            # should hydrate be calculated using the dry basis composition ?
            fracs = self.GetPort(IN_PORT).GetCompositionValues()
            value, self.hydrateStatus = thAdmin.GetSpecialProperty(prov, case, option, fracs, 'HYDRATECURVE')

            # Make sure we did not get an empty value
            try:
                valIsEmpty = (float(value) == -12321)
            except:
                valIsEmpty = 0

            try:
                if valIsEmpty:
                    self.hydrateResults = None
                    if not self.IsForgetting():
                        # hydMsg += self.hydrateStatus + '\n'
                        self.InfoMessage('ThermoProviderMsg', (self.GetPath(), self.hydrateStatus), addToUnitOpMsg=1)
                else:
                    valsList = value.split()
                    rows, cols = len(valsList) / 2, 2
                    vals = list(map(float, valsList))[:-1]
                    vals = np.reshape(np.array(vals, dtype=float), (rows, cols))
                    self.hydrateResults = vals
            except:
                self.hydrateResults = None
                if not self.IsForgetting():
                    # hydMsg += self.hydrateStatus + '\n'
                    self.InfoMessage('ThermoProviderMsg', (self.GetPath(), self.hydrateStatus), addToUnitOpMsg=1)

        # if envMsg and hydMsg:
        # self.unitOpMessage = ('ThermoProviderMsg', (self.GetPath(), '%s\n%s' %(envMsg, hydMsg)))
        # elif envMsg:
        # self.unitOpMessage = ('RawOutput', envMsg)
        # elif hydMsg:
        # self.unitOpMessage = ('ThermoProviderMsg', (self.GetPath(), hydMsg))

        # TH Envelope
        if self.thEnvelope:
            self.thEnvelope.Solve()
        # PH Envelope
        if self.phEnvelope:
            self.phEnvelope.Solve()
        return

    def GetComposition(self, thAdmin, prov, case):
        # returns None if composition is not ready
        # otherwise, optionally strip out water
        vals = []
        cmps = self.GetPort(IN_PORT).GetCompositionValues()
        dryBasis = self.GetParameterValue(DRYBASIS_PAR)

        for i in range(len(cmps)):
            vals.append(0.0)

        envCmps = thAdmin.GetArrayProperty(prov, case, (T_VAR, 1.0), (P_VAR, 1.0),
                                           LIQUID_PHASE, vals, 'EnvelopeCmp')

        sum_cmp = 0.0
        for i in range(len(cmps)):
            if cmps[i] is None:
                return None
            if dryBasis and cmps[i] > 0.0 and envCmps[i] == 0:
                cmps[i] = 0.0
                self._strippedH2O = 1  # i have removed water
            sum_cmp += cmps[i]

        if sum_cmp == 0.0:
            if len(cmps) > 0:
                self.unitOpMessage = ('RawOutput', 'No valid compound after stripping water')
            return None

        for i in range(len(cmps)):
            cmps[i] /= sum_cmp

        return cmps

    def IsWater(self, idx):

        # obsolete method

        # i may need a better way to identify water with different thermo provider
        # Perhaps a property call IsWater is needed?
        thCaseObj = self.GetThermo()
        thAdmin, prov, case = thCaseObj.thermoAdmin, thCaseObj.provider, thCaseObj.case
        formula = thAdmin.GetSelectedCompoundProperties(prov, case, idx, 'Formula')

        return formula[0] == 'H2O' or formula[0] == 'h2o' or formula[0] == 'HOH' or formula[0] == 'hoh'

    def ForgetAllCalculations(self):
        super(PTEnvelope, self).ForgetAllCalculations()
        for line in list(self.QualityLines.values()):
            line.SetResults(None)
        self.hydrateResults = None
        self.hydrateStatus = ''

    def SetParameterValue(self, paramName, value):
        if paramName == INITP_PAR:
            q1 = QualityCurve(1.0)
            self.AddObject(q1, 'Q1')
            port = self.GetPort(STARTING_P_PORT)
            port.SetValue(500.0, FIXED_V)
            vals = value.split()
            self.pSet.SetValues(vals)
        else:
            super(PTEnvelope, self).SetParameterValue(paramName, value)

    def FreeCurveName(self, name):
        # The quality curves must be unique, even with the TH curves
        baseName = name
        for i in range(1000):
            obj = self.GetObject(name)
            if obj is None:
                return name
            name = baseName + '_' + str(i)
        return name


class THEnvelope(object):
    def __init__(self):
        self.QualityLines = {}
        self.ptEnvelope = None
        self.name = ''

    def Initialize(self, ptEnvelope, name):
        self.name = name
        self.ptEnvelope = ptEnvelope
        # create the default saturation curve with name = SATRUATED_CURVE and Pspec<0
        sat = QualityCurve(-1)
        self.AddObject(sat, name + SATRUATED_CURVE)

    def CleanUp(self):
        self.ptEnvelope = None
        for c in list(self.QualityLines.values()):
            c.CleanUp()
        self.QualityLines.clear()

    def GetObject(self, name):
        if name in self.QualityLines:
            return self.QualityLines[name]

    def GetContents(self):
        t = re.sub(' .*', '', repr(self))[1:]
        paths = t.split('.')
        path = paths[len(paths) - 1]
        result = [(path + 'Object', self.name)]
        sortedKeys = list(self.QualityLines.keys())[:]
        sortedKeys.sort(lambda a, b: cmp(self.QualityLines[a].vapFrac, self.QualityLines[b].vapFrac))
        result.append((path + ENTHALPYCURVES, sortedKeys))
        for key in list(self.QualityLines.keys()):
            result.append((self.QualityLines[key].name, self.QualityLines[key]))
        return result

    def AddObject(self, obj, name):
        if isinstance(obj, QualityCurve):
            name = self.ptEnvelope.FreeCurveName(name)
            obj.Initialize(self, name)
            self.QualityLines[name] = obj
            self.ptEnvelope.PushSolveOp(self.ptEnvelope)

    def DeleteObject(self, obj):
        if isinstance(obj, QualityCurve):
            name = obj.name
            if name in self.QualityLines:
                obj.CleanUp()
                del self.QualityLines[name]
                self.ptEnvelope.PushSolveOp(self.ptEnvelope)

    def ForgetAllCalculations(self):
        for line in list(self.QualityLines.values()):
            line.SetResults(None)

    def Solve(self):
        # The ptEnvelope must have been solved first and the saturation line exists
        saturationLine = None
        # first get the saturation line
        for line in list(self.ptEnvelope.QualityLines.values()):
            if line.vapFrac == 0.0 or line.vapFrac == 1.0:
                saturationLine = line.results
                break
        if saturationLine is None:
            return

        thCaseObj = self.ptEnvelope.GetThermo()
        thAdmin, prov, case = thCaseObj.thermoAdmin, thCaseObj.provider, thCaseObj.case
        cmps = self.ptEnvelope.GetComposition(thAdmin, prov, case)
        compounds = CompoundList(None)
        compounds.SetLocalCompValues(cmps)

        for line in list(self.QualityLines.values()):
            if not cmps or thCaseObj is None:
                # i am not ready, remove the existing result (if any)
                line.SetResults(None)
            else:
                typeList = copy.deepcopy(saturationLine.pointTypes)
                pList = copy.deepcopy(saturationLine.pValues)
                tList = copy.deepcopy(saturationLine.tValues)
                pSpec = line.vapFrac
                # sort the temperature list for the isobars
                n = saturationLine.pointCount
                if pSpec > 0:
                    tList.sort()
                    # add 5 extra data points
                    n += 5
                    t0 = tList[len(tList) - 1]
                    delt = 0.04 * t0
                    pList.extend((pSpec, pSpec, pSpec, pSpec, pSpec))
                    tList.extend((t0 + delt, t0 + 2 * delt, t0 + 3 * delt, t0 + 4 * delt, t0 + 5 * delt))
                    typeList.extend((0, 0, 0, 0, 0))

                hVals = []
                props = MaterialPropertyDict()
                for i in range(n):
                    if pSpec < 0:
                        t = tList[i]  # Saturation line, must be the first curve in the list
                        p = pList[i]
                    else:
                        p = pSpec  # isobar
                        t = tList[i]
                        pList[i] = p
                        typeList[i] = 0
                    props[P_VAR].SetValue(p, FIXED_V)
                    props[T_VAR].SetValue(t, FIXED_V)

                    flashResults = thAdmin.Flash(prov, case, compounds, props, 1, (H_VAR,))
                    # store H in the temperature slot, K-values = None
                    hVals.append(flashResults.bulkProps[0])
                envResults = EnvelopeResults(0, '', n, typeList, pList, tList, hVals, THCURVE_INDICATOR)
                line.SetResults(envResults)

    def SetParent(self, parent):
        self.ptEnvelope = parent

    def Clone(self):
        clone = self.__class__()
        clone.name = self.name
        for qcName in self.QualityLines:
            clone.QualityLines[qcName] = self.QualityLines[qcName].Clone()
            clone.QualityLines[qcName].SetParent(clone)
        return clone


class PHEnvelope(THEnvelope):
    def Solve(self):
        # The ptEnvelope must have been solved first and the saturation line exists
        saturationLine = None
        # first get the saturation line
        for line in list(self.ptEnvelope.QualityLines.values()):
            if line.vapFrac == 0.0 or line.vapFrac == 1.0:
                saturationLine = line.results
                break
        if saturationLine is None:
            return

        thCaseObj = self.ptEnvelope.GetThermo()
        thAdmin, prov, case = thCaseObj.thermoAdmin, thCaseObj.provider, thCaseObj.case
        cmps = self.ptEnvelope.GetComposition(thAdmin, prov, case)
        compounds = CompoundList(None)
        compounds.SetLocalCompValues(cmps)

        for line in list(self.QualityLines.values()):
            if not cmps or thCaseObj is None:
                # i am not ready, remove the existing result (if any)
                line.SetResults(None)
            else:
                typeList = copy.deepcopy(saturationLine.pointTypes)
                pList = copy.deepcopy(saturationLine.pValues)
                tList = copy.deepcopy(saturationLine.tValues)
                tSpec = line.vapFrac
                # sort the pressure list for the isotherm
                n = saturationLine.pointCount
                if tSpec > 0:
                    pList.sort()
                    # add 5 extra data points
                    n += 5
                    p0 = pList[len(pList) - 1]
                    delp = 0.04 * p0
                    pList.extend((p0 + delp, p0 + 2 * delp, p0 + 3 * delp, p0 + 4 * delp, p0 + 5 * delp))
                    tList.extend((tSpec, tSpec, tSpec, tSpec, tSpec))
                    typeList.extend((0, 0, 0, 0, 0))

                hVals = []
                props = MaterialPropertyDict()

                for i in range(n):
                    if tSpec < 0:
                        t = tList[i]  # Saturation line, must be the first curve in the list
                        p = pList[i]
                    else:
                        t = tSpec  # isotherm
                        p = pList[i]
                        tList[i] = t
                        typeList[i] = 0
                    props[P_VAR].SetValue(p, FIXED_V)
                    props[T_VAR].SetValue(t, FIXED_V)

                    flashResults = thAdmin.Flash(prov, case, compounds, props, 1, (H_VAR,))
                    # store H in the temperature slot, K-values = None
                    hVals.append(flashResults.bulkProps[0])
                envResults = EnvelopeResults(0, '', n, typeList, pList, tList, hVals, PHCURVE_INDICATOR)
                line.SetResults(envResults)
