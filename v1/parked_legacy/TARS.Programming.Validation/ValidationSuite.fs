module TARS.Programming.Validation.ValidationSuite

open TARS.Programming.Validation.ProgrammingLearning
open TARS.Programming.Validation.MetascriptEvolution
open TARS.Programming.Validation.AutonomousImprovement
open TARS.Programming.Validation.ProductionIntegration

type ValidationResult =
    { ProgrammingLearning: bool
      MetascriptEvolution: bool
      AutonomousImprovement: float
      ProductionIntegration: bool }

let runSuite () =
    let learningValidator = ProgrammingLearningValidator()
    let evolutionValidator = MetascriptEvolutionValidator()
    let improvementValidator = AutonomousImprovementValidator()
    let productionValidator = ProductionIntegrationValidator()

    let programmingLearning = learningValidator.RunValidation()
    let evolutionPassed = evolutionValidator.ValidateEvolution()
    let improvementSummary = improvementValidator.Validate()
    let productionPassed =
        productionValidator.ValidateProductionDeployment()
        && productionValidator.ValidateTarsCLIIntegration()
        && productionValidator.ValidateFLUXIntegration()
        && productionValidator.ValidateBlueGreenEnvironment()
        && productionValidator.ValidateMonitoringSystem()

    { ProgrammingLearning = programmingLearning
      MetascriptEvolution = evolutionPassed
      AutonomousImprovement = improvementSummary.ImprovementScore
      ProductionIntegration = productionPassed }

