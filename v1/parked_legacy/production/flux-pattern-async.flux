PATTERN async_workflow {
    let asyncBind f m = async {
        let! result = m
        return! f result
    }
}