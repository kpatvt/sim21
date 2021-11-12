def Messages():
    """create dictionary of English messages"""
    m = {'AddCompoundError': "Thermo provider reports the following error when adding compound:\n%s",
         'AdjustingFromOlderVersion': "Recalling case created in an older version. Updating from: FlowsheetVersion = "
                                      "%d; ReleaseVersion = %s. To: FlowsheetVersion %d; ReleaseVersion %s",
         'AfterPortDisconnect': "%s disconnected from %s",
         'BalanceInvalidPort': "Invalid port for balance (not material or energy)",
         'BeforePortDisconnect': "Disconnecting %s from %s",
         'BubbleTCouldNotCalc': "Bubble Point temperature could not be calculated in %s at P = %s kPa and composition "
                                "= %s",
         'CalcDisturbance': "Calculating disturbance %i of %i in jacobian of %s",
         'CalculatingProfile': "Calculating profile in %s. Segment %i. Properties %s",
         'CalculatingStep': "Calculating step %i in %s. Currently in %g. Going from %g to %g",
         'CantAddObject': "Can't add %s to %s", 'CantAddToStage': "Can't add %s to stage %d of %s",
         'CantAddToStageObject': "Can't add %s to %s on stage %d of %s", 'CantChangeName': "Can't change name of %s",
         'CantCloneFlowsheet': "Can't clone flowsheet %s if stacks are not empty (solve, forget, unconverged "
                               "recycles, consistency errors)",
         'CantCreateSpec': "Can't create spec %s. It is probably not supported",
         'CantDeleteFromStage': "Can't delete %s from stage %d of %s",
         'CantDeleteObject': "Can't delete object %s. Unit op can not solve with out it",
         'CantDelPortDirectly': "Can't delete port %s from %s. Delete associated object instead",
         'CantEstimate': "Could not estimate missing %s while initializing %s",
         'CantFindPhCh': "Can't find phase changes in %s for more than two sides or when solving in rating mode (UA "
                         "values specified)",
         'CantMoveToStage': "Can't move %s to stage %d of %s. Make sure there are no conflicting names",
         'CantOverwriteThermo': "Can't overwrite a thermo case. The correct procedure is to first delete old thermo "
                                "and then set a new thermo. Unit op: %s; Current thermo: %s",
         'CantSetIP': "Can't set interaction parameter with value %f for compounds %s and %s",
         'CantSetLiqPhPar': "Can't set number of liquid phases to %s",
         'CantSetSingleFrac': "Can't set the mass or volume fraction of one single compound %s in a material port %s.",
         'CantSetParameter': "Can't set parameter %s to value %s",
         'CantUseSpecInZeroFlow': "Can't use specs in a zero flow draw %s.",
         'ChangedEffMatrix': "The efficiencies matrix changed as a result of a change in configuration in %s",
         'ChangedPortState': "Changed state of port %s to %d (0=Normal port; 1=Recycle port)",
         'CompNotNormalized': "Mole fractions of %s sums to %f, not 1",
         'ConnectErrorNoPort': "Can't connect %s.%s to %s.%s as a port is missing",
         'ConnectErrorNoUop': "Can't connect %s.%s to %s.%s as a unit op is missing",
         'ConnectSameTypePorts': "Attempt to connect ports of differing types in %s",
         'ConnectSigToNonSig': "Attempt to connect signal port %s to a non signal port",
         'ContDerivCalc': "Controller solver for %s calculating derivative %d",
         'ControllerConvergeFail': "Controller solver for %s failed to converge",
         'ControllerTotalError': "Controller solver for %s error - %f", 'Converged': "Converged %s in %i iterations",
         'ConvergedOp': "Converged %s", 'CouldNotConverge': "Could not converge %s after %d iterations",
         'CouldNotConvergeInner': "Could not converge Inner loop %s after %d iterations",
         'CouldNotConvergeOuter': "Could not converge Outer loop %s after %d iterations",
         'CouldNotConvergeUA': "Could not solve for UA = %f in %s",
         'CouldNotInitialize': "Could not initialize set of equations when solving %s",
         'CouldNotInvertJacobian': "Could not invert Jacobian in %s",
         'CouldNotLoadLanguage': "Could not load language %s",
         'CouldNotLoadProvider': "Could not load thermo provider %s",
         'CouldNotRestorePlugIn': "Could not restore plug in object %s when recalling case. The default object will "
                                  "be used instead",
         'CouldNotSolve': "Could not solve %s",
         'CouldNotSolveNonSuppFlash': "Could not solve non supported flash with variables %s = %s, %s = %s in %s",
         'CreatePortTypeError': "Port %s does not have a valid type in %s",
         'CrossConnMoleLoss': "A significant loss of mole flow of %f was detected in the cross connector %s. A common "
                              "reason is the mismatch of compounds that contain significant flows",
         'DeletePortError': "Cannot delete port %s from %s",
         'DewTCouldNotCalc': "Dew Point temperature could not be calculated in %s at P = %s kPa and composition = %s",
         'DiffThCaseInConn': "Different thermo case found across port connection %s -> %s. The values could not be "
                             "passed",
         'DoneProfile': "Done calculating profile in %s",
         'DuplicateName': "Command failed due to a duplication of the name %s in %s",
         'ErrInCleanUp': "Error while cleaning up %s",
         'ErrNotifyChangeCmp': "Error while notifying %s of a change in the compounds list",
         'ErrNotifyLiqChange': "Error while notifying %s of a change of the number of liquid phases. LiquidPhases = %s",
         'ErrNotifyParChange': "Error while notifying %s of a change of the value of a parameter. %s = %s",
         'ErrNotifySolChange': "Error while notifying %s of a change of the number of solid phases. LiquidPhases = %s",
         'ErrNotifyThChange': "Error while notifying %s of a change of thermodynamic case. ThermoCase = %s",
         'ERRSettingThermo': "Error attempting to set thermo into unit op: %s; Thermo attempted: %s",
         'ErrSpecialProp': "Error calculating special property in %s. Message form thermo provider: %s",
         'ErrorSolvingDesign': "Error solving design object %s",
         'ERRTowerArithmetic': "Tower %s failed to converge due to an arithmetic error",
         'EqnCalcError': "Calculation error in %s",
         'EqnDuplicateSigName': "Signal name %s is used more than once in equation %s",
         'EqnNumbMismatch': "Error in equation counting in %s",
         'EqnParenMismatch': "Mismatched parenthesis in %s of Equation %s",
         'EqnSyntax': "Syntax error in %s in Equation %s",
         'EqnUnknownToken': "Don't know how to deal with %s in equation %s of %s",
         'EqnBasedUOpError': "%s Iteration %d Max Error %f",
         'FlashFailure': "Flash failed in %s. Message from Thermo Provider: %s",
         'HotTLowerThanColdT': "The temperature of the hot inlet %f is  lower than the temperature of the cold inlet "
                               "%f in %s",
         'HydrateCouldNotCalc': "Hydrate temperature could not be calculated in %s at P = %s kPa and composition = %s",
         'HydrateLowP': "Hydrate can not be formed at low pressure condition of P = %s kPa in %s",
         'InnerErrorDetail': "%s Inner Details. Error: %13.6g ; MaxErrorValue: %13.6g ; MaxErrorEqnName: %s ",
         'InnerLoopSummary': """%s Inner Loop Summary:
        MaxErrorEqnName:......... %s
        MaxErrorValue:........... %.6g

        MaxDeltaTStage(0 at top): %i
        MaxDeltaTValue(New-Old):. %.4g

        Converged:............... %i
        Iterations:.............. %i""", 'InvalidCalcStatusInSet': "Invalid calcStatus in SetValue",
         'InvalidComposition': "The %s composition = %f in %s.  It has been reset to zero.",
         'InvalidDrawPhase': "Invalid phase for draw on stage %d of %s",
         'InvalidTowerSpecPhase': "Invalid phase in spec on stage %d of %s",
         'LumpLiqs': "A second liquid with fraction %f is detected in a two phase VL flash.",
         'MaxSolverIterExceeded': "Maximum %d iterations exceeded in solving flowsheet %s",
         'MissingSpecs': "Missing %d specifications", 'MissingVariable': "Missing %s in %s",
         'MissigZInCommonProps': "Z Factor should always be in the common properties. Attempted to set: %s",
         'NonHydrateFormerFound': "Non hydrate former was found coming into %s",
         'NoPortDirection': "Port %s requires direction (in or out) in %s",
         'NoSupportForReqArrProps': "The thermo provider %s doesn't support the following required array properties %s",
         'NoSupportForReqProps': "The thermo provider %s doesn't support the following required properties %s",
         'NotConverging': "%s does not seem to be converging and calculations were stopped. Change the parameter "
                          "MonitorConvergence to 0 if you wish to deactivate this feature",
         'NoVersionUpdate': "No update for %d (%s) to %d (%s)",
         'ODEMaxSteps': "Maximum integration steps reached (%i) in %s. Increase ODEMaxSteps if integration was "
                        "proceeding correctly",
         'OuterErrorDetail': "%s Iteration %d Outer Error %13.6g. MaxErrorStage(0 at top) %i WaterDrawError %13.6g",
         'OverspecFlash': "Could not perform flash calculation in %s because it is overspecified. Only 2 variables "
                          "needed and %i were given (%s)",
         'PortNotFlashedDesignObj': "Ports from unit op are not flashed therefore design object %s not ready to be "
                                    "solved",
         'RawOutput': "%s", 'RecycleErrorDetail': "%s %s %g vs %g",
         'RecycleConsistency': "Consistency Error %s %s %g vs %g", 'RecycleIter': "Iteration %d -> max Error %f in %s",
         'RenamePort': "Rename port %s.%s to %s.  It is connected to %s",
         'RenamePortError': "Cannot rename port %s to %s",
         'RenamePortNameExists': "Cannot rename port %s to %s as that name is already used",
         'RevertingFromNewerVersion': "Recalling case created in a newer version. Updating from: flowsheet version "
                                      "%d, release version %s. To: flowsheet version %d release version %s",
         'SetValueUnknownNotNone': "SetValue with UNKNOWN_V flag must have value = None",
         'SetVarTypeMismatch': "Port variable type %s is not %s in %s",
         'SigConnectTypeMismatch': "Variable type conflict (%s vs %s) when connecting %s to %s",
         'SigShareMismatch': "Variable type conflict (%s vs %s) when sharing %s with %s",
         'SolvingDesign': "Solving design object %s", 'SolvingOp': "Solving operation %s",
         'SpecConflict': "Specification conflict between %s and %s in %s", 'Status': "%s",
         'StepSizeTooSmall': "Step size underflow in %s. Step size = %g",
         'TemperatureCross': "Temperature cross (%f %f) in %s",
         'InternalTCross': "Internal temperature cross in %s. See profiles for details",
         'NoPkgSelected': "No thermo package was selected when attempted to create %s",
         'ThermoProviderMsg': "Msg from thermo provider when solving %s:\n%s",
         'TooManySolidPhases': "Too many solid phases requested(%d) when attempting flash from %s",
         'TooManyTowerSpecs': "%d specs found, only %d needed in %s",
         'TowerCalcJacobian': "Calculating Jacobian for %s",
         'TowerCmpMatrixError': "%s had an error in solving the material balances for component %d",
         'TowerDeletePort': "Cannot directly delete port %s from %s. Select and delete the associated draw or spec",
         'TowerEffSetToOne': "Tower efficiency in the top stage was set to 1.0 because the vapour draw is 0",
         'TowerFailedToConverge': "%s failed to converge in %d iterations - error = %f",
         'TowerInnerError': "%s Inner Error %f", 'TowerNoPressure': "No outlet pressures available for tower %s",
         'TowerOuterError': "%s Iteration %d Outer Error %f", 'TowerQSpecError': "Can't assign energy flow to stage %d",
         'TowerRemoveLastStage': "Cannot remove %d stages from below stage %d",
         'TowerPARemovalError': "Cannot remove a stage with a feed from a pump around unless the pump around is "
                                "removed too. Feed is in stage %i, pump around from stage %i",
         'TowerSSRemoveError': "Top or bottom tower stages cannot be removed unless the whole section is removed",
         'TowerUpdateEffErr': "An error occurred while attempting to update the efficiencies matrix in %s. Please "
                              "update manually",
         'TowerMissingFeedInfo': "Feed %s is not fully specified", 'TwrNoFeed': "No feeds were found in %s",
         'TwrSpecErr': "Error while calculating the spec %s",
         'TwrSpecErrConfig': "The spec %s was installed into an invalid object %s. For example, a pump around spec "
                             "installed into something different from a pump around",
         'TwrSubCooledVapDraw': "Tower failed to converge due to a sub cooled solution at the top where there is a "
                                "vapour draw. Degrees of subcooling = %f",
         'UnresolvedConsistencyErrors': "The following consistency errors in flowsheet %s have not been resolved ("
                                        "only lists one per unit operation):\n%s",
         'UnresolvedRecycles': "The following recycle ports in flowsheet %s have not been converged (only lists one "
                               "per unit operation):\n%s",
         'UpdateInvalidPort': "Port %s does not exist in %s - can't update",
         'WrongDiamEjector': "Wrong diameter specification in %s. Nozzle diameter must be smaller than throat "
                             "diameter. Nozzle D = %f; Throat D = %f",
         'WrongNumberTowerSpecs': "Mismatch in number of tower specs - %d vs %d needed in %s",
         'WrongParentDesignObj': "Design object %s contained in the wrong type of unit operation",
         'WrongSetting': "Invalid value %s for setting %s in object %s", 'DoneSolving': "Flowsheet %s solved",
         'NoMessage': "", 'MissingValue': "%s has no value", 'ErrorValue': "Error = %s", 'OK': "OK", 'T': "Temperature",
         'P': "Pressure", 'H': "Enthalpy", 'VapFrac': "VapFrac", 'MoleFlow': "MoleFlow", 'MassFlow': "MassFlow",
         'VolumeFlow': "VolumeFlow", 'Energy': "Energy", 'MolecularWeight': "MolecularWeight", 'ZFactor': "ZFactor"}

    # Following messages not in alphabetical order to keep all the properties together
    return m
