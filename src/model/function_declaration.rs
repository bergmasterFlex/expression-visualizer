#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct FunctionDeclarationId(pub usize);

#[derive(Clone)]
pub struct FunctionDeclaration {
    pub name: String,
    pub inputs: Vec<FunctionParameterDeclaration>,
    pub output_type: crate::infer::EType,
}

#[derive(Clone)]
pub struct FunctionParameterDeclaration {
    pub name: String,
    /// The type this parameter accepts, or `None` when it accepts any value.
    ///
    /// `None` is the *absence of a constraint*, not the `None` type: `=` and
    /// `!=` compare two values of any kinds, and the language has no top type
    /// to say that with — `EType` must not grow one. The checker already reads
    /// a missing anchor type as "unconstrained", so an unconstrained parameter
    /// needs no further handling there.
    pub r#type: Option<crate::infer::EType>,
}

/// The defined function set, normative source: thesis appendix
/// `app:defined-functions`.
///
/// Written as a table rather than as literal structs: twenty-two declarations
/// spelled out field by field run to several hundred lines in which the one
/// thing that matters — the signature — is the hardest part to read.
///
/// A declaration's id is its row index, so ids stay stable as long as rows are
/// appended rather than inserted.
///
/// The partial functions carry their failure in the output type (`Integer |
/// None`) instead of failing: the sad path is a value that travels along an
/// edge, which is what makes the caller match it out.
pub fn catalogue() -> std::collections::HashMap<FunctionDeclarationId, FunctionDeclaration> {
    let integer = || crate::infer::EType::Int(None);
    let boolean = || crate::infer::EType::Bool(None);
    let string = || crate::infer::EType::String(None);
    let character = || crate::infer::EType::Char(None);
    let text = || crate::infer::EType::SumType(vec![character(), string()]);
    let or_none =
        |t: crate::infer::EType| crate::infer::EType::SumType(vec![t, crate::infer::EType::None]);
    // Unconstrained: see `FunctionParameterDeclaration::type`.
    let any = || None;
    let int2 = |a: &'static str, b: &'static str| vec![(a, Some(integer())), (b, Some(integer()))];

    let table: Vec<(
        &str,
        Vec<(&str, Option<crate::infer::EType>)>,
        crate::infer::EType,
    )> = vec![
        // Arithmetic
        ("+", int2("summand1", "summand2"), integer()),
        ("-", int2("minuend", "subtrahend"), integer()),
        ("*", int2("factor1", "factor2"), integer()),
        ("/", int2("dividend", "divisor"), or_none(integer())),
        ("mod", int2("dividend", "divisor"), or_none(integer())),
        ("neg", vec![("number", Some(integer()))], integer()),
        // Comparison. `=` and `!=` are total over every pair of values, so
        // they constrain neither parameter.
        ("=", vec![("left", any()), ("right", any())], boolean()),
        ("!=", vec![("left", any()), ("right", any())], boolean()),
        ("<", int2("left", "right"), boolean()),
        (">", int2("left", "right"), boolean()),
        ("<=", int2("left", "right"), boolean()),
        (">=", int2("left", "right"), boolean()),
        // Logic. Neither `&&` nor `||` short-circuits: a function call asks
        // for every argument, so both operands are evaluated.
        (
            "&&",
            vec![("left", Some(boolean())), ("right", Some(boolean()))],
            boolean(),
        ),
        (
            "||",
            vec![("left", Some(boolean())), ("right", Some(boolean()))],
            boolean(),
        ),
        ("!", vec![("operand", Some(boolean()))], boolean()),
        // String
        ("len", vec![("str", Some(string()))], integer()),
        (
            "charAt",
            vec![("str", Some(string())), ("i", Some(integer()))],
            or_none(character()),
        ),
        (
            "concat",
            vec![("left", Some(text())), ("right", Some(text()))],
            string(),
        ),
        (
            "substr",
            vec![
                ("str", Some(string())),
                ("begin", Some(integer())),
                ("length", Some(integer())),
            ],
            or_none(string()),
        ),
        // Math
        ("min", int2("left", "right"), integer()),
        ("max", int2("left", "right"), integer()),
        ("abs", vec![("number", Some(integer()))], integer()),
    ];

    table
        .into_iter()
        .enumerate()
        .map(|(index, (name, inputs, output_type))| {
            (
                FunctionDeclarationId(index),
                FunctionDeclaration {
                    name: name.to_string(),
                    inputs: inputs
                        .into_iter()
                        .map(|(name, r#type)| FunctionParameterDeclaration {
                            name: name.to_string(),
                            r#type,
                        })
                        .collect(),
                    output_type,
                },
            )
        })
        .collect()
}
