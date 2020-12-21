"""Models a Conversion reactor

Classes:
ReactionDisplay - For rendering reaction
ConversionReaction - Conversion reaction class
IsothermalConvReactor - Iso-thermal conversion reactor class
ConvReactor - General conversion reactor, containing:
              an isothermal conversion reactor
              a heater
              an energy balance
"""
import re

from sim21.solver.Messages import MessageHandler
from sim21.unitop import UnitOperations, Balance, Heater, Sensor, Stream
from sim21.solver import Error
from sim21.solver.Variables import *

# Reactor constants
from sim21.unitop.BaseForReactors import QEXOTHERMIC_ISPOS_PAR, NURXN_PAR, REACTION, COEFF
from sim21.unitop.BaseForReactors import RXNFFORMULA_PAR, RXNCONV, RXNEXTENT

from sim21.kludges import cmp

# Particular constants for this unit op
BALANCEDRXN = 'BalancedRxn'
RXNORDER_PAR = 'RxnOrder'
SIMULTANEOUSRXN_PAR = 'SimultaneousRxn'


class ReactionDisplay(object):
    """ An object to display the reaction detail """

    def __init__(self, parent):
        self.parent = parent
        self.exportList = []

    def __str__(self):
        rxn = self.parent
        result = 'Reaction = ' + rxn.FormulaString() + '\nOrder = ' + \
                 str(rxn.GetParameterValue(RXNORDER_PAR)) + '\nStoichmetric coefficients:'
        maxLength = 0
        for cmpName in rxn.cmpNames: maxLength = max(maxLength, len(cmpName))
        for i in range(len(rxn.stoichCoeffs)):
            result += '\n   ' + rxn.cmpNames[i] + ' ' * (maxLength - len(rxn.cmpNames[i]) + 2)
            v = rxn.stoichCoeffs[i]
            if v < 0:
                result += '%f' % v
            else:
                result += ' %f' % v
            if i == rxn.baseCompIdx: result += ' (Base Comp)'
        return result

    def CleanUp(self):
        self.parent = None

    def GetValues(self):
        rxn = self.parent
        self.exportList = []
        self.exportList.append(rxn.rxnName)
        self.exportList.append(rxn.baseCompIdx)
        self.exportList.append(self.parent.RxnConv())
        self.exportList.append(rxn.GetParameterValue(RXNORDER_PAR))
        self.exportList.extend(rxn.stoichCoeffs)

        return self.exportList


class ConversionReaction(UnitOperations.UnitOperation):
    """ class for a conversion reaction """

    def __init__(self, initScript=None):
        UnitOperations.UnitOperation.__init__(self, initScript)
        self.rxnName = 'Unknown'
        self.rxnExtent = BasicProperty(GENERIC_VAR)  # so that i can export for inspection
        self.rxnExtent.SetValue(0.0)
        self.stoichCoeffs = []
        self.cmpNames = []
        self.baseCompIdx = -1
        self.rxnConv = None

        self.SetParameterValue(RXNFFORMULA_PAR, '')
        self.SetParameterValue(RXNORDER_PAR, -1)
        self.display = ReactionDisplay(self)

    def SetParameterValue(self, paramName, value):
        UnitOperations.UnitOperation.SetParameterValue(self, paramName, value)
        if paramName == RXNFFORMULA_PAR and self.parameters[paramName] != '':
            self.ParseFormula()
            # stored the parsed formula to ger rid of the compound index in the formula
            parsedFormula = self.FormulaString()
            super(ConversionReaction, self).SetParameterValue(paramName, parsedFormula)

    def CleanUp(self):
        if self.display:
            self.display.CleanUp()
            self.display = None
        self.rxnConv = None
        super(ConversionReaction, self).CleanUp()

    def GetContents(self):
        result = [('Reaction', self.display)]
        return result

    def GetObject(self, name):
        if name == COEFF:
            return self.display
        elif name == RXNEXTENT:  # export for display
            return self.rxnExtent
        elif name == RXNCONV:
            return self.rxnConv
        elif name == BALANCEDRXN:
            return self.BalancedRxn()
        else:
            return super(ConversionReaction, self).GetObject(name)

    def FormulaString(self):
        # Build the formula from internal data
        # Group the product first; then the reactant
        cmpNames = self.GetCompoundNames()
        formula = ''
        for loop in range(2):
            for i in range(len(cmpNames)):
                if i == self.baseCompIdx:
                    cmp0 = "!'" + cmpNames[i] + "'"
                else:
                    cmp0 = "'" + cmpNames[i] + "'"
                cmp0 = re.sub(' ', '_', cmp0)
                coeff = self.Coeff(i)
                if loop == 0 and coeff > 0.0:
                    formula += '+' + str(coeff) + '*' + cmp0
                elif loop == 1 and coeff < 0:
                    formula += str(coeff) + '*' + cmp0
        return self.rxnName + ':' + formula[1:]

    def ParseFormula(self):
        eqnStr = self.parameters[RXNFFORMULA_PAR]
        eqn = eqnStr  # keep a copy of the original equation
        if eqnStr is None or eqnStr == '': return

        # reset all coeffs to zero
        cmpNames = self.GetCompoundNames()
        self.cmpNames = cmpNames

        self.stoichCoeffs = []
        for i in range(len(cmpNames)):
            self.stoichCoeffs.append(0)

        # replace compounds within quotes by the index
        # for compounds with '-'
        cmps = re.findall(r'"[^"]+"|\'[^\']+\'', eqnStr)
        for token in cmps:
            # strip out the quote
            cmp0 = token[1:-1]
            # underscore represent space
            cmp0 = re.sub('_', ' ', cmp0)
            try:
                idx = cmpNames.index(cmp0)
                eqnStr = re.sub(token, str(idx), eqnStr)
            except:
                pass

        try:
            # extract the reaction name
            tokens = re.split(r':', eqnStr, 1)
            if len(tokens) == 2:
                eqnStr = tokens[1].strip()
                self.rxnName = tokens[0].strip()

            # replace all - by +- so that when i split the tokens,
            # the signs of the coeff are preserved
            eqnStr = re.sub('-', '+-', eqnStr)
            tokens = re.split(r'\+', eqnStr)
            for token in tokens:
                if token.strip() == '':
                    continue
                x = re.split(r'\*', token.strip())
                # if coeff is missing, assume 1 or -1
                if len(x) == 1:
                    x0 = x[0]
                    if x0[0] == '-':
                        x.append(x0[1:])
                        x[0] = '-1'
                    else:
                        x.append(x0)
                        x[0] = '1'
                # let underscores stand for spaces
                cmp0 = re.sub('_', ' ', x[1].strip())
                # base compound indicator
                baseCmp = 0
                if cmp0[0] == '!':
                    cmp0 = cmp0[1:]
                    baseCmp = 1
                # if the input compound name is numberic, it is the compound index
                try:
                    idx = int(cmp0)
                except:
                    idx = cmpNames.index(cmp0)
                coef = float(x[0].strip())
                self.stoichCoeffs[idx] = coef
                if baseCmp:
                    self.baseCompIdx = idx
        except:
            # self.SetParameterValue(RXNFFORMULA_PAR, '')
            self.stoichCoeffs = []
            raise Error.SimError('EqnSyntax', (eqn, self.GetPath()))

        # base compound must be a reactant
        # check for equation mass balance later (need MW of selected compounds)
        if self.baseCompIdx < 0:
            raise Error.SimError('EqnSyntax', (eqn, self.GetPath()))
        elif self.stoichCoeffs[self.baseCompIdx] >= 0:
            raise Error.SimError('EqnSyntax', (eqn, self.GetPath()))

    def Coeff(self, ith):
        try:
            return self.stoichCoeffs[ith]
        except:
            return 0.0

    def RxnConv(self):
        try:
            x = self.rxnConv.GetValue()
            x = min(x, 1.0)
            x = max(x, 0.0)
            return x
        except:
            return 0.0

    def BaseCompIdx(self):
        return self.baseCompIdx

    def AppendCompound(self, cmpIdx=-1):
        """Add a compound """
        super(ConversionReaction, self).AppendCompound(cmpIdx)
        try:
            self.ParseFormula()
        except:
            eqn = self.GetParameterValue(RXNFFORMULA_PAR)
            self.InfoMessage('EqnSyntax', (eqn, self.GetPath()))

    def AfterCompoundDeleted(self, cmpName):
        """Deletes a compound from the reaction"""
        super(ConversionReaction, self).AfterCompoundDeleted(cmpName)
        # reparse the formula to get rid of the references to the deleted compound
        try:
            self.SetParameterValue(RXNFFORMULA_PAR, self.parameters[RXNFFORMULA_PAR])
        except:
            eqn = self.GetParameterValue(RXNFFORMULA_PAR)
            self.InfoMessage('EqnSyntax', (eqn, self.GetPath()))

    def MoveCompound(self, cmp1Idx, cmp2Idx):
        super(ConversionReaction, self).MoveCompound(cmp1Idx, cmp2Idx)
        # formula must not be using the compound index
        try:
            self.ParseFormula()
        except:
            eqn = self.GetParameterValue(RXNFFORMULA_PAR)
            self.InfoMessage('EqnSyntax', (eqn, self.GetPath()))

    def ThermoChanged(self, thCaseObj):
        """
        intercept this to set up the stoichCoeffs list
        """
        # YK to do: fix up later to match compound names
        super(ConversionReaction, self).ThermoChanged(thCaseObj)
        try:
            self.ParseFormula()
        except:
            eqn = self.GetParameterValue(RXNFFORMULA_PAR)
            self.InfoMessage('EqnSyntax', (eqn, self.GetPath()))

    def ValidateOk(self):
        if self.baseCompIdx < 0: return 0
        eqn = self.parameters[RXNFFORMULA_PAR]
        if eqn is None or eqn == '': return 0
        cmpNames = self.GetCompoundNames()
        if len(cmpNames) > 0 and len(self.stoichCoeffs) == 0: return 0
        if self.rxnConv:
            if self.rxnConv.GetValue() is None: return 0
        return 1

    def BalancedRxn(self):
        # Check the MW to see if the reaction is balanced
        # returns 1 if balanced, 0 if not and None if not sure
        try:
            lhs = 0.0
            rhs = 0.0
            thCaseObj = self.GetThermo()
            thAdmin, prov, case = thCaseObj.thermoAdmin, thCaseObj.provider, thCaseObj.case
            for i in range(len(self.cmpNames)):
                mw = thAdmin.GetSelectedCompoundProperties(prov, case, i, [MOLE_WT])[0]
                coeff = self.stoichCoeffs[i]
                if coeff > 1.0e-6:
                    lhs = lhs + coeff * mw
                elif coeff < -1.0e-6:
                    rhs = rhs - coeff * mw
            if abs(rhs - lhs) < 1.0e-3:
                return 1
            else:
                return 0
        except:
            return None

    def _CloneParameters(self, clone, attrNamesToClone):
        # Clone parameters
        for paramName in self.parameters:
            # Do a copy just in case
            clone.parameters[paramName] = copy.deepcopy(self.parameters[paramName])

        for paramName in self.parameterPropertyTypes:
            # Can safely point to the same thing as they are global types
            clone.parameterPropertyTypes[paramName] = self.parameterPropertyTypes[paramName]

        return attrNamesToClone


class IsothermalConvReactor(UnitOperations.UnitOperation):
    def __init__(self, initScript=None):
        UnitOperations.UnitOperation.__init__(self, initScript)
        self.ProductTotalMole = 0.0
        self.feedMolei = []
        self.productMolei = []
        self.productMoleFrac = []
        self.inPort = self.CreatePort(MAT | IN, IN_PORT)
        self.outPort = self.CreatePort(MAT | OUT, OUT_PORT)

        self.qPort = self.CreatePort(ENE | OUT, OUT_PORT + 'Q')

        self.containerUnitOp = None  # used by containing op to be notified of conversion changes

        self.SetParameterValue(NURXN_PAR, 0)
        self.SetParameterValue(SIMULTANEOUSRXN_PAR, 1)
        self.trace = 1.0e-10

    def CleanUp(self):
        self.inPort = self.outPort = self.qPort = self.containerUnitOp = None
        super(IsothermalConvReactor, self).CleanUp()

    def GetListOfReqParam(self):
        return NURXN_PAR

    def SetParameterValue(self, paramName, value):

        super(IsothermalConvReactor, self).SetParameterValue(paramName, value)

        if paramName == NURXN_PAR: self.UpdateRxnCount()

    def UpdateRxnCount(self):
        """Update the amount and names of the ports in"""
        nuRxns = len(self.GetChildUnitOps())
        rxnIn = self.parameters[NURXN_PAR]

        for i in range(nuRxns, rxnIn, -1):
            self.DelUnitOperation(REACTION + str(i - 1))
            self.DeletePortNamed(REACTION + str(i - 1) + '_' + RXNCONV)
        for i in range(nuRxns, rxnIn):
            rxn = ConversionReaction()
            rxn.SetParameterValue(RXNORDER_PAR, i + 1)
            self.AddUnitOperation(rxn, REACTION + str(i))
            # create conversion signal port
            port = self.CreatePort(SIG, REACTION + str(i) + '_' + RXNCONV)
            port.SetSignalType(FRAC_VAR)  # fration
            port.SetValue(1.0, FIXED_V)  # default is total conversion
            rxn.rxnConv = port  # set the reaction's signal port
        # notify my container
        if self.containerUnitOp:
            self.containerUnitOp.GrabConversionPorts()

    def RemoveRxn(self, name):
        self.DelUnitOperation(name)

    def CalculateProduct(self, finalCalc=0):
        # if finalCalc, then reset negative trcce to zero
        rxns = list(self.chUODict.values())
        sumX = 0.0
        totFeed = sum(self.feedMolei)

        # Do not make the compound flows = to 0.0 if the overall flow is tiny already
        checkForTraces = abs(totFeed) > self.trace

        for m in range(self.nc):
            dum = 0.0
            for aRxn in rxns:
                dum = dum + aRxn.rxnExtent.GetValue() * aRxn.Coeff(m)

            pdt = self.feedMolei[m] + dum
            if finalCalc:
                # just to prevent round offs
                if abs(pdt) < self.trace:
                    if checkForTraces or pdt < 0.0:
                        pdt = 0.0
                elif pdt < 0.0:
                    if abs(pdt) > self.trace:
                        cmpName = str(self.GetCompoundNames()[m])
                        self.InfoMessage('InvalidComposition', (cmpName, pdt, self.GetPath()))
                    pdt = 0.0
            self.productMolei[m] = pdt

            sumX = sumX + pdt
        # normalize new moles
        self.ProductTotalMole = sumX
        self.productMoleFrac = self.productMolei / sumX

    def PropagateValue(self, var):
        inPort = self.inPort
        outPort = self.outPort
        inVal = inPort.GetPropValue(var)
        outVal = outPort.GetPropValue(var)
        if inVal is not None and outVal is None: outPort.SetPropValue(var, inVal, CALCULATED_V)
        if outVal is not None and inVal is None: inPort.SetPropValue(var, outVal, CALCULATED_V)

    def Solve(self):
        # if SIMULTANEOUSRXN_PAR is not defined, assume 1

        # If zero flow, then just pass the zero and quit
        if self.inPort.GetPropValue(MOLEFLOW_VAR) == 0.0:
            self.outPort.SetPropValue(MOLEFLOW_VAR, 0.0, CALCULATED_V)
            return
        elif self.outPort.GetPropValue(MOLEFLOW_VAR) == 0.0:
            self.inPort.SetPropValue(MOLEFLOW_VAR, 0.0, CALCULATED_V)
            return

        # Pick solve method
        result = self.parameters.get(SIMULTANEOUSRXN_PAR, 1)
        if result == 0:
            retVal = self.SolveSequential()
        else:
            retVal = self.SolveSimultaneous()

        # Mass balance check for inconsistencies
        massFlow = self.inPort.GetPropValue(MASSFLOW_VAR)
        if massFlow is not None:
            self.outPort.SetPropValue(MASSFLOW_VAR, massFlow, CALCULATED_V)

        return retVal

    def SolveSimultaneous(self):
        if self.IsForgetting(): return 0
        inPort = self.inPort
        outPort = self.outPort
        # propagate pressure
        self.PropagateValue(P_VAR)
        self.PropagateValue(T_VAR)
        # flash the inlet
        self.FlashAllPorts()
        # Propagate again in case something new got known
        self.PropagateValue(P_VAR)
        self.PropagateValue(T_VAR)

        try:
            if not self.ValidateOk(): return 0
            rxns = list(self.chUODict.values())
            xin = inPort.GetCompositionValues()
            self.nc = len(xin)

            # define the working arrays
            self.feedMolei = np.array(xin)
            self.productMolei = np.zeros(self.nc, dtype=float)
            self.productMoleFrac = np.zeros(self.nc, dtype=float)
            # Initialize rxn extents.  If inlet flow is unknown, assume 1 mole
            flow = inPort.GetPropValue(MOLEFLOW_VAR)
            if flow is None:  # and not self.IsForgetting():
                flow = 1.0
            self.feedMolei = self.feedMolei * flow

            for aRxn in rxns:
                idx = aRxn.BaseCompIdx()
                conv = aRxn.RxnConv()
                ext = -conv * self.feedMolei[idx] / aRxn.Coeff(idx)
                aRxn.rxnExtent.SetValue(ext)
                aRxn.localExt = ext
            self.CalculateProduct()
        except:
            return 0

        scheme = 1
        for i in range(self.nc):
            if self.productMolei[i] < 0.0:
                # Find the scaling on the reaction conversion for this component
                stoichSum = 0.0
                if scheme == 0:
                    for aRxn in rxns:
                        if aRxn.rxnExtent.GetValue() > self.trace:
                            stoichSum = stoichSum + aRxn.Coeff(i)
                    scaling = self.productMolei[i] / stoichSum

                    for aRxn in rxns:
                        rxnExtent = aRxn.rxnExtent.GetValue()
                        if aRxn.Coeff(i) < 0.0 and rxnExtent > self.trace:
                            newExtent = rxnExtent - scaling
                            if abs(newExtent) < abs(aRxn.localExt):
                                aRxn.localExt = newExtent
                else:
                    for aRxn in rxns:
                        if aRxn.rxnExtent.GetValue() > self.trace:
                            stoichSum = stoichSum + aRxn.Coeff(i) * aRxn.rxnExtent.GetValue()
                    scaling = self.productMolei[i] / stoichSum
                    factor = 1.0 - self.productMolei[i] / stoichSum
                    for aRxn in rxns:
                        rxnExtent = aRxn.rxnExtent.GetValue()
                        if aRxn.Coeff(i) < 0.0 and rxnExtent > self.trace:
                            newExtent = rxnExtent * factor
                            aRxn.localExt = newExtent

        for aRxn in rxns:
            aRxn.rxnExtent.SetValue(aRxn.localExt)
        self.CalculateProduct(1)
        self.UpdateOutPort()
        return 1

    def UpdateOutPort(self):
        # composition
        sumX = 0.0
        for m in range(self.nc):
            sumX = sumX + self.productMoleFrac[m]
        self.productMoleFrac = self.productMoleFrac / sumX

        self.outPort.SetCompositionValues(self.productMoleFrac, CALCULATED_V)
        # molar flow
        if self.inPort.GetPropValue(MOLEFLOW_VAR) is not None:
            # inlet flow is known, assign outlet
            self.outPort.SetPropValue(MOLEFLOW_VAR, self.ProductTotalMole, CALCULATED_V)
        elif self.outPort.GetPropValue(MOLEFLOW_VAR) is not None:
            # outlet is known, calculate was done by assuming 1 mole inlet
            # proportionate the inlet flow
            inMoleFlow = self.outPort.GetPropValue(MOLEFLOW_VAR) / self.ProductTotalMole
            self.inPort.SetPropValue(MOLEFLOW_VAR, inMoleFlow, CALCULATED_V)
            self.inPort.CalcFlows()
        # flash the port
        self.outPort.Flash()
        self.EnergyBalance()

    def ValidateOk(self):
        rxns = list(self.chUODict.values())
        # all reactions must have been defined
        for aRxn in rxns:
            if not aRxn.ValidateOk(): return 0
        # fix 020503, do not check for complete inlet stream,
        # check for inlet composiiton
        if not self.inPort.GetCompounds().AreValuesReady(): return 0

        # flow must have been known
        # inPort = self.inPort
        # flow = inPort.GetPropValue(MOLEFLOW_VAR)
        # if flow == None: return 0
        # The In port must have been fully defined.  Probably should be OK if P is missing
        # if not inPort.AlreadyFlashed(): return 0
        return 1

    def EnergyBalance(self):
        # i need the H and flow for both the inlet and outlet
        flowIn = self.inPort.GetPropValue(MOLEFLOW_VAR)
        flowOut = self.outPort.GetPropValue(MOLEFLOW_VAR)

        if flowIn is not None and flowOut is not None:
            hrxnIn = self.RxnEnthalpy(self.inPort)
            hrxnOut = self.RxnEnthalpy(self.outPort)
            if hrxnIn is not None and hrxnOut is not None:
                hout = (hrxnIn * flowIn - hrxnOut * flowOut) / 3.6  # convert from KJ/hr -> W (=J/s)
                self.qPort.SetPropValue(ENERGY_VAR, hout, CALCULATED_V)

    def RxnEnthalpy(self, aPort):
        h = aPort.GetPropValue(H_VAR)
        if h is None: return None
        try:
            p = aPort.GetPropValue(P_VAR)
            t = aPort.GetPropValue(T_VAR)
            thCaseObj = self.GetThermo()
            thAdmin = thCaseObj.thermoAdmin
            provider = thCaseObj.provider
            thName = thCaseObj.case
            prop1 = (P_VAR, p)
            prop2 = (T_VAR, t)
            phase = LIQUID_PHASE
            frac = aPort.GetCompositionValues()
            propList = ('rxnBaseH', MOLE_WT)
            value = thAdmin.GetProperties(provider, thName, prop1, prop2, phase, frac, propList)
            hrxn = h + value[0]
            return hrxn
        except:
            return None

    def SolveSequential(self):
        if self.IsForgetting(): return 0
        inPort = self.inPort
        outPort = self.outPort

        # propagate pressure
        self.PropagateValue(P_VAR)
        self.PropagateValue(T_VAR)
        # flash the inlet
        self.FlashAllPorts()
        # Propagate again in case something new got known
        self.PropagateValue(P_VAR)
        self.PropagateValue(T_VAR)

        try:
            if not self.ValidateOk(): return 0
            rxns = list(self.chUODict.values())
            xin = inPort.GetCompositionValues()
            self.nc = len(xin)

            # define the working arrays
            self.feedMolei = np.array(xin)
            self.productMolei = np.zeros(self.nc, dtype=float)
            self.productMoleFrac = np.zeros(self.nc, dtype=float)
            # Initialize rxn extents.  If inlet flow is unknown, assume 1 mole
            flow = inPort.GetPropValue(MOLEFLOW_VAR)
            if flow is None:  # and not self.IsForgetting():
                flow = 1.0
            self.feedMolei = self.feedMolei * flow
            # initialize the product for cases where there is no reactions
            self.productMoleFrac = self.feedMolei
            self.ProductTotalMole = inPort.GetPropValue(MOLEFLOW_VAR)

            # sort the rxns according to reaction order
            rxns.sort(lambda a, b: cmp(a.GetParameterValue(RXNORDER_PAR), b.GetParameterValue(RXNORDER_PAR)))

            # react one reaction at a time accoring the the specified reaction order
            for aRxn in rxns:
                idx = aRxn.BaseCompIdx()
                conv = aRxn.RxnConv()
                ext = -conv * self.feedMolei[idx] / aRxn.Coeff(idx)
                aRxn.rxnExtent.SetValue(ext)
                self.CalculateProductForReaction(aRxn)

                factor = 1.0
                for i in range(self.nc):
                    if self.productMolei[i] < 0.0:
                        # Find the scaling on the reaction conversion for this component
                        if abs(ext) > self.trace:
                            factor0 = 1.0 - self.productMolei[i] / (aRxn.Coeff(i) * ext)
                            if factor0 < factor:
                                factor = factor0
                        else:
                            self.productMolei[i] = 0.0

                aRxn.rxnExtent.SetValue(ext * factor)
                self.CalculateProductForReaction(aRxn, 1)

                # reset the feed to be the product after a reaction
                self.feedMolei = self.productMolei * 1.0

            self.UpdateOutPort()
            return 1
        except:
            return 0

    def CalculateProductForReaction(self, aRxn, finalCalc=0):
        # if finalCalc, then reset negative trcce to zero
        rxns = list(self.chUODict.values())
        sumX = 0  #
        for m in range(self.nc):
            dum = aRxn.rxnExtent.GetValue() * aRxn.Coeff(m)

            pdt = self.feedMolei[m] + dum
            if finalCalc:
                # just to prevent round offs
                if abs(pdt) < self.trace:
                    pdt = 0.0
                elif pdt < 0.0:
                    if abs(pdt) > self.trace:
                        cmpName = str(self.GetCompoundNames()[m])
                        self.InfoMessage('InvalidComposition', (cmpName, pdt, self.GetPath()))
                    pdt = 0.0
            self.productMolei[m] = pdt

            sumX = sumX + pdt
        # normalize new moles
        self.ProductTotalMole = sumX
        self.productMoleFrac = self.productMolei / sumX


class ConvReactor(UnitOperations.UnitOperation):
    # This version do not suuport full backward calc with the following known probelms:
    #    The inlet composition cannot be calculated from the outlet composition
    #    Given OutQ and Out.T, it cannot calculate In.T
    def __init__(self, initScript=None):
        super(ConvReactor, self).__init__(initScript)
        self.isoRxn = IsothermalConvReactor()
        self.AddUnitOperation(self.isoRxn, 'IsoRxn')
        self.isoRxn.containerUnitOp = self

        self.heater = Heater.Heater()
        self.AddUnitOperation(self.heater, 'RxnHeater')

        self.baln = Balance.BalanceOp()
        self.AddUnitOperation(self.baln, 'EneBalance')

        # Initialize with one energy In and Out as a default
        # The parameter QEXOTHERMIC_ISPOS_PAR will set the missing energy port
        self.baln.SetParameterValue(NUSTIN_PAR + Balance.S_ENE, 1)
        self.baln.SetParameterValue(NUSTOUT_PAR + Balance.S_ENE, 1)
        self.baln.SetParameterValue(Balance.BALANCETYPE_PAR, Balance.ENERGY_BALANCE)

        # to create an Energy Out port for export
        self.balEneSensor = Sensor.EnergySensor()
        self.AddUnitOperation(self.balEneSensor, 'BalEneSensor')
        self.eneSensor = Sensor.EnergySensor()
        self.AddUnitOperation(self.eneSensor, 'EneSensor')
        self.eneStream = Stream.Stream_Energy()
        self.AddUnitOperation(self.eneStream, 'EneStream')

        # connect the child unit ops
        self.ConnectPorts('IsoRxn', OUT_PORT + 'Q', 'EneBalance', IN_PORT + 'Q0')
        self.ConnectPorts('RxnHeater', IN_PORT + 'Q', 'EneBalance', OUT_PORT + 'Q0')
        self.ConnectPorts('IsoRxn', OUT_PORT, 'RxnHeater', IN_PORT)

        self.ConnectPorts('EneSensor', OUT_PORT, 'EneStream', IN_PORT)
        #
        self.ConnectPorts('BalEneSensor', SIG_PORT, 'EneSensor', SIG_PORT)

        # borrow child ports
        self.BorrowChildPort(self.eneStream.GetPort(OUT_PORT), OUT_PORT + 'Q')
        self.BorrowChildPort(self.isoRxn.GetPort(IN_PORT), IN_PORT)
        self.BorrowChildPort(self.heater.GetPort(OUT_PORT), OUT_PORT)
        self.BorrowChildPort(self.heater.GetPort(DELTAP_PORT), DELTAP_PORT)

        self.SetParameterValue(NURXN_PAR, 0)
        self.SetParameterValue(SIMULTANEOUSRXN_PAR, 1)
        self.SetParameterValue(QEXOTHERMIC_ISPOS_PAR, 1)

    def CleanUp(self):
        self.isoRxn = self.heater = self.baln = None
        self.balEneSensor = self.eneSensor = self.eneStream = None
        super(ConvReactor, self).CleanUp()

    def SetParameterValue(self, paramName, value):

        if paramName == QEXOTHERMIC_ISPOS_PAR and value == self.parameters.get(QEXOTHERMIC_ISPOS_PAR, None):
            return

        super(ConvReactor, self).SetParameterValue(paramName, value)

        if paramName == NURXN_PAR:
            self.isoRxn.SetParameterValue(paramName, value)
        elif paramName == SIMULTANEOUSRXN_PAR:
            self.isoRxn.SetParameterValue(paramName, value)
        elif paramName == QEXOTHERMIC_ISPOS_PAR:
            if value:
                self.baln.SetParameterValue(NUSTIN_PAR + Balance.S_ENE, 1)
                self.baln.SetParameterValue(NUSTOUT_PAR + Balance.S_ENE, 2)
                self.ConnectPorts('BalEneSensor', IN_PORT, 'EneBalance', OUT_PORT + 'Q1')
            else:
                self.baln.SetParameterValue(NUSTIN_PAR + Balance.S_ENE, 2)
                self.baln.SetParameterValue(NUSTOUT_PAR + Balance.S_ENE, 1)
                self.ConnectPorts('BalEneSensor', OUT_PORT, 'EneBalance', IN_PORT + 'Q1')

    def ValidateParameter(self, paramName, value):
        if not super(ConvReactor, self).ValidateParameter(paramName, value):
            return False

        if paramName == QEXOTHERMIC_ISPOS_PAR:
            if value != 0 and value != 1:
                return False

        return True

    def DeleteObject(self, obj):
        if isinstance(obj, UnitOperations.OpParameter):
            if obj.GetName() == QEXOTHERMIC_ISPOS_PAR:
                self.InfoMessage('CantDeleteObject', (obj.GetPath(),), MessageHandler.errorMessage)
                return

        super(ConvReactor, self).DeleteObject(obj)

    def GetObject(self, name):
        obj = super(ConvReactor, self).GetObject(name)
        if not obj:
            obj = self.isoRxn.GetObject(name)
        return obj

    def GrabConversionPorts(self):
        """
        Borrow the iso-thermal reactor conversion ports
        """
        # clear all signal port
        self.ports_sig.clear()
        # restore the delP signal port
        self.BorrowChildPort(self.heater.GetPort(DELTAP_PORT), DELTAP_PORT)
        # add the conversion port
        ports = self.isoRxn.GetPorts(SIG)
        for port in ports:
            self.BorrowChildPort(port, port.GetName())
        UnitOperations.UnitOperation.SetParameterValue(self, NURXN_PAR, len(ports))
