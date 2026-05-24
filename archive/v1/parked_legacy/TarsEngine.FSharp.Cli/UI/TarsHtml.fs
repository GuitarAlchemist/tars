namespace TarsEngine.FSharp.Cli.UI

/// Real Elmish Virtual DOM with Message Dispatching
module TarsHtml =
    type VNode =
        | Element of string * (string * obj) list * VNode list
        | Text of string
        | Empty

    and EventHandler<'Msg> = 'Msg

    let div attrs children = Element("div", attrs, children)
    let h1 attrs children = Element("h1", attrs, children)
    let h2 attrs children = Element("h2", attrs, children)
    let h3 attrs children = Element("h3", attrs, children)
    let h4 attrs children = Element("h4", attrs, children)
    let span attrs children = Element("span", attrs, children)
    let button attrs children = Element("button", attrs, children)
    let ul attrs children = Element("ul", attrs, children)
    let li attrs children = Element("li", attrs, children)
    let text str = Text(str)
    let empty = Empty

    // Event attribute helpers for REAL JavaScript event dispatching
    let onClick (jsFunction: string) = ("onclick", box jsFunction)
    let onInput (jsFunction: string) = ("oninput", box jsFunction)
    let className (cls: string) = ("class", box cls)
    let style (s: string) = ("style", box s)

    let rec render (node: VNode) =
        match node with
        | Text content -> content
        | Empty -> ""
        | Element(tag, attrs, children) ->
            let attrString =
                attrs
                |> List.map (fun (key, value) ->
                    match value with
                    | :? string as s -> sprintf "%s=\"%s\"" key s
                    | _ -> sprintf "%s=\"%A\"" key value)
                |> String.concat " "
            let childrenString =
                children
                |> List.map render
                |> String.concat ""

            if List.isEmpty children then
                sprintf "<%s %s />" tag attrString
            else
                sprintf "<%s %s>%s</%s>" tag attrString childrenString tag
