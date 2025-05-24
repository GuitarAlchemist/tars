namespace TarsEngine.FSharp.Core

/// Module containing collection utilities
module Collections =
    /// Module containing List utilities
    module List =
        /// Returns the first element that satisfies the predicate, or None if no element satisfies the predicate
        let tryFind predicate list =
            list |> List.tryFind predicate

        /// Returns the first element that satisfies the predicate, or the default value if no element satisfies the predicate
        let findOrDefault defaultValue predicate list =
            match list |> List.tryFind predicate with
            | Some x -> x
            | None -> defaultValue

        /// Returns a new list with distinct elements based on a key selector
        let distinctBy keySelector list =
            list
            |> List.fold (fun (acc, keys) x ->
                let key = keySelector x
                if Set.contains key keys then
                    (acc, keys)
                else
                    (x :: acc, Set.add key keys)
            ) ([], Set.empty)
            |> fst
            |> List.rev

        /// Chunks a list into sublists of the specified size
        let chunk size list =
            let rec loop acc remaining =
                match remaining with
                | [] -> List.rev acc
                | _ ->
                    let chunk, rest = remaining |> List.splitAt (min size (List.length remaining))
                    loop (chunk :: acc) rest
            loop [] list

        /// Splits a list into two lists based on a predicate
        let partition predicate list =
            list |> List.partition predicate

        /// Returns a new list with elements that appear in both lists
        let intersect list1 list2 =
            list1 |> List.filter (fun x -> List.contains x list2)

        /// Returns a new list with elements that appear in the first list but not in the second
        let difference list1 list2 =
            list1 |> List.filter (fun x -> not (List.contains x list2))

        /// Returns a new list with elements that appear in either list, but not both
        let symmetricDifference list1 list2 =
            let diff1 = difference list1 list2
            let diff2 = difference list2 list1
            diff1 @ diff2

        /// Returns a new list with elements from both lists, removing duplicates
        let union list1 list2 =
            list1 @ (difference list2 list1)

        /// Returns a new list with elements from the first list that satisfy the predicate
        let filterMap predicate mapper list =
            list
            |> List.filter predicate
            |> List.map mapper

        /// Returns a new list with elements from the first list, applying the function if the predicate is satisfied
        let mapIf predicate mapper list =
            list |> List.map (fun x -> if predicate x then mapper x else x)

        /// Returns a new list with elements from the first list, applying the first function if the predicate is satisfied, otherwise applying the second function
        let mapIfElse predicate ifMapper elseMapper list =
            list |> List.map (fun x -> if predicate x then ifMapper x else elseMapper x)

    /// Module containing Map utilities
    module Map =
        /// Returns the value associated with the key, or None if the key is not found
        let tryFind key map =
            map |> Map.tryFind key

        /// Returns the value associated with the key, or the default value if the key is not found
        let findOrDefault defaultValue key map =
            match map |> Map.tryFind key with
            | Some x -> x
            | None -> defaultValue

        /// Returns a new map with the key-value pair added or updated
        let addOrUpdate key value map =
            map |> Map.add key value

        /// Returns a new map with the key-value pair added if the key does not exist
        let addIfNotExists key value map =
            if map |> Map.containsKey key then
                map
            else
                map |> Map.add key value

        /// Returns a new map with the key-value pair updated if the key exists
        let updateIfExists key value map =
            if map |> Map.containsKey key then
                map |> Map.add key value
            else
                map

        /// Returns a new map with the key-value pairs from both maps, using the second map's value if the key exists in both
        let merge map1 map2 =
            Map.fold (fun acc key value -> Map.add key value acc) map1 map2

        /// Returns a new map with the key-value pairs from both maps, using the provided function to resolve conflicts
        let mergeWith resolver map1 map2 =
            Map.fold (fun acc key value ->
                if Map.containsKey key acc then
                    Map.add key (resolver key (Map.find key acc) value) acc
                else
                    Map.add key value acc
            ) map1 map2

        /// Returns a new map with the key-value pairs that satisfy the predicate
        let filterMap predicate mapper map =
            map
            |> Map.filter predicate
            |> Map.map (fun _ v -> mapper v)

        /// Returns a new map with the keys mapped using the provided function
        let mapKeys mapper map =
            map
            |> Map.toSeq
            |> Seq.map (fun (k, v) -> (mapper k, v))
            |> Map.ofSeq

    /// Module containing Set utilities
    module Set =
        /// Returns a new set with elements that appear in both sets
        let intersect set1 set2 =
            Set.intersect set1 set2

        /// Returns a new set with elements that appear in the first set but not in the second
        let difference set1 set2 =
            Set.difference set1 set2

        /// Returns a new set with elements that appear in either set, but not both
        let symmetricDifference set1 set2 =
            let diff1 = difference set1 set2
            let diff2 = difference set2 set1
            Set.union diff1 diff2

        /// Returns a new set with elements from both sets
        let union set1 set2 =
            Set.union set1 set2

        /// Returns a new set with elements from the first set that satisfy the predicate
        let filterMap predicate mapper set =
            set
            |> Set.filter predicate
            |> Set.map mapper
