let rec reverseList lst = 
    match lst with
        | [] -> []
        | head :: tail -> tail @ [head]