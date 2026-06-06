let rec reverseList lst =
    match lst with
    | [] -> []
    | x :: xs -> reverseList xs @ [x]