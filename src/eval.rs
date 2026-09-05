#[derive(Clone)]
pub struct State {
    pub node_ids_to_values: std::collections::HashMap<crate::model::node::Id, EValue>,
}

#[derive(Clone)]
pub enum EValue {
    Bool(bool),
    Int(i32),
    String(String),
    Char(char),
    None(String),
}

impl State {
    fn empty() -> Self {
        Self {
            node_ids_to_values: std::collections::HashMap::new(),
        }
    }

    pub fn new(
        ast: &crate::model::ast::Ast,
        user_vardecl_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<Self, Vec<String>> {
        ast.nodes
            .get(&ast.sink_node_id)
            .cloned()
            .ok_or(vec![format!("sink node {} not found", ast.sink_node_id)])
            .and_then(|sink_node| {
                Self::empty().eval_next_step(
                    ast,
                    user_vardecl_values,
                    (ast.sink_node_id.clone(), sink_node),
                    function_declarations,
                )
            })
    }

    pub fn eval_next_step(
        &self,
        ast: &crate::model::ast::Ast,
        user_vardecl_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        (visitor_node_id, visitor_node): (crate::model::node::Id, crate::model::node::ENode),
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<Self, Vec<String>> {
        if self.node_ids_to_values.contains_key(&visitor_node_id) {
            Ok(self.clone())
        } else {
            let input_anchor_ids_to_node_ids =
                ast.get_connected_nodes_to_node_input_anchors(&visitor_node_id);
            let input_anchor_ids_to_values = input_anchor_ids_to_node_ids
                .iter()
                .filter_map(|(anchor_id, node_id)| {
                    self.node_ids_to_values
                        .get(node_id)
                        .map(|value| (anchor_id.clone(), value.clone()))
                })
                .collect::<std::collections::HashMap<_, _>>();
            if input_anchor_ids_to_node_ids.len() == input_anchor_ids_to_values.len() {
                self.eval_value_for_node(
                    &visitor_node_id,
                    visitor_node,
                    input_anchor_ids_to_values,
                    user_vardecl_values,
                    ast,
                    function_declarations,
                )
            } else {
                let next_step_sub_evals = input_anchor_ids_to_node_ids
                    .into_iter()
                    .map(|(_, node_id)| match ast.nodes.get(&node_id).cloned() {
                        Some(node) => self.eval_next_step(
                            ast,
                            user_vardecl_values,
                            (node_id, node),
                            function_declarations,
                        ),
                        None => Err(vec![format!("node {} not found", node_id)]),
                    })
                    .collect::<Vec<Result<Self, Vec<String>>>>();
                let errors = next_step_sub_evals
                    .iter()
                    .filter_map(|result| result.as_ref().err())
                    .flatten()
                    .cloned()
                    .collect::<Vec<String>>();
                if errors.is_empty() {
                    Ok(next_step_sub_evals
                        .into_iter()
                        .filter_map(Result::ok)
                        .fold(self.clone(), Self::merged_with))
                } else {
                    Err(errors)
                }
            }
        }
    }

    fn with_value(&self, node_id: &crate::model::node::Id, value: EValue) -> Self {
        Self {
            node_ids_to_values: self
                .node_ids_to_values
                .clone()
                .into_iter()
                .chain(vec![(node_id.clone(), value)])
                .collect(),
        }
    }

    fn merged_with(self, other: Self) -> Self {
        Self {
            node_ids_to_values: self
                .node_ids_to_values
                .into_iter()
                .chain(other.node_ids_to_values)
                .collect(),
        }
    }

    pub fn eval_value_for_node(
        &self,
        node_id: &crate::model::node::Id,
        node: crate::model::node::ENode,
        input_anchor_ids_to_values: std::collections::HashMap<crate::model::anchor::Id, EValue>,
        user_vardecl_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        ast: &crate::model::ast::Ast,
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<Self, Vec<String>> {
        match node {
            crate::model::node::ENode::Program {} => {
                Err(vec!["cannot get value of a program node".to_string()])
            }
            crate::model::node::ENode::Sink { input_anchor } => input_anchor_ids_to_values
                .get(&input_anchor)
                .cloned()
                .map(|value| self.with_value(node_id, value))
                .ok_or_else(|| vec!["no value found for input anchor for sink".to_string()]),
            crate::model::node::ENode::FunctionCall {
                function_declaration_id,
                input_anchors,
                ..
            } => {
                let mut sorted_input_anchors = input_anchors.clone();
                sorted_input_anchors.sort_by_key(|anchor_id| match ast.anchors.get(anchor_id) {
                    Some(crate::model::anchor::EAnchor::Input(input_anchor)) => {
                        input_anchor.order_num
                    }
                    _ => usize::MAX,
                });
                sorted_input_anchors
                    .iter()
                    .map(|anchor_id| {
                        input_anchor_ids_to_values
                            .get(anchor_id)
                            .cloned()
                            .ok_or(format!("no value found for input anchor {:?}", anchor_id))
                    })
                    .collect::<Result<Vec<EValue>, String>>()
                    .and_then(|arguments| {
                        function_declarations
                            .get(&function_declaration_id)
                            .ok_or(format!(
                                "function declaration with id {} not found",
                                function_declaration_id.0
                            ))
                            .and_then(|function_declaration| {
                                Self::eval_value_for_function_call(function_declaration, arguments)
                            })
                    })
                    .map(|value| self.with_value(node_id, value))
                    .map_err(|error| vec![error])
            }
            crate::model::node::ENode::ConstDecl { r#type, .. } => {
                Self::eval_value_for_type(r#type)
                    .map(|value| self.with_value(node_id, value))
                    .map_err(|error| vec![error])
            }
            crate::model::node::ENode::TypeCast {
                r#type,
                input_anchor,
                ..
            } => input_anchor_ids_to_values
                .get(&input_anchor)
                .ok_or_else(|| vec!["no value found for input anchor for type cast".to_string()])
                .map(|value| {
                    self.with_value(
                        node_id,
                        Self::eval_value_for_type_cast(value.clone(), r#type),
                    )
                }),
            crate::model::node::ENode::VarDecl { name, .. } => user_vardecl_values
                .get(node_id)
                .cloned()
                .map(|value| self.with_value(node_id, value))
                .ok_or_else(|| vec![format!("no value for var-decl provided: {}", name)]),
            crate::model::node::ENode::Match {
                patterns,
                input_anchor,
                ..
            } => input_anchor_ids_to_values
                .get(&input_anchor)
                .cloned()
                .ok_or_else(|| vec!["no value found for input anchor for match".to_string()])
                .and_then(|input_value| {
                    Self::eval_pattern_match(patterns, ast, input_value)
                        .map_err(|error| vec![error])
                })
                .and_then(
                    |matching_pattern_id| match ast.nodes.get(&matching_pattern_id) {
                        Some(crate::model::node::ENode::Pattern { sink_node_id, .. }) => match self
                            .node_ids_to_values
                            .get(sink_node_id)
                        {
                            Some(sink_value) => Ok(self.with_value(node_id, sink_value.clone())),
                            None => match ast.nodes.get(sink_node_id).cloned() {
                                Some(sink_node) => self.eval_next_step(
                                    ast,
                                    user_vardecl_values,
                                    (sink_node_id.clone(), sink_node),
                                    function_declarations,
                                ),
                                None => Err(vec![format!("node {} not found", sink_node_id)]),
                            },
                        },
                        Some(_) => Err(vec![format!(
                            "matched node {} is not a pattern",
                            matching_pattern_id
                        )]),
                        None => Err(vec![format!("node {} not found", matching_pattern_id)]),
                    },
                ),
            crate::model::node::ENode::Pattern { parent_match, .. } => {
                match ast.nodes.get(&parent_match) {
                    Some(crate::model::node::ENode::Match { input_anchor, .. }) => ast
                        .get_connected_nodes_to_anchor(input_anchor.clone())
                        .into_iter()
                        .find_map(|source_node_id| {
                            self.node_ids_to_values.get(&source_node_id).cloned()
                        })
                        .map(|value| self.with_value(node_id, value))
                        .ok_or_else(|| {
                            vec![format!(
                                "no value at parent match input anchor for pattern {}",
                                node_id
                            )]
                        }),
                    Some(_) => Err(vec![format!(
                        "parent match {} of pattern {} is not a match node",
                        parent_match, node_id
                    )]),
                    None => Err(vec![format!("parent match {} not found", parent_match)]),
                }
            }
        }
    }

    fn eval_pattern_match(
        patterns: Vec<crate::model::node::Id>,
        ast: &crate::model::ast::Ast,
        input_value: EValue,
    ) -> Result<crate::model::node::Id, String> {
        patterns
            .into_iter()
            .find(|pattern_id| match ast.nodes.get(pattern_id) {
                Some(crate::model::node::ENode::Pattern { r#type, .. }) => {
                    Self::value_matches_type(&input_value, r#type)
                }
                _ => false,
            })
            .ok_or_else(|| format!("no pattern matched value {}", input_value))
    }

    fn value_matches_type(
        input_value: &EValue,
        pattern_type: &crate::model::r#type::EType,
    ) -> bool {
        match (pattern_type, input_value) {
            (crate::model::r#type::EType::Any, _) => true,
            (crate::model::r#type::EType::Bool { value }, EValue::Bool(b)) => value
                .as_ref()
                .is_none_or(|v| v.parse::<bool>().ok() == Some(*b)),
            (crate::model::r#type::EType::Int { value }, EValue::Int(i)) => value
                .as_ref()
                .is_none_or(|v| v.parse::<i32>().ok() == Some(*i)),
            (crate::model::r#type::EType::String { value }, EValue::String(s)) => {
                value.as_ref().is_none_or(|v| v == s)
            }
            (crate::model::r#type::EType::Char { value }, EValue::Char(c)) => value
                .as_ref()
                .is_none_or(|v| v.parse::<char>().ok() == Some(*c)),
            (crate::model::r#type::EType::None { .. }, EValue::None(_)) => true,
            _ => false,
        }
    }

    fn eval_value_for_type(r#type: crate::model::r#type::EType) -> Result<EValue, String> {
        match r#type {
            crate::model::r#type::EType::Bool { value } => value
                .ok_or("bool type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<bool>()
                        .map(EValue::Bool)
                        .map_err(|_| format!("could not parse \"{}\" as bool", v))
                }),
            crate::model::r#type::EType::Int { value } => value
                .ok_or("int type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<i32>()
                        .map(EValue::Int)
                        .map_err(|_| format!("could not parse \"{}\" as int", v))
                }),
            crate::model::r#type::EType::String { value } => value
                .map(EValue::String)
                .ok_or("string type did not have a specific value!".to_string()),
            crate::model::r#type::EType::Char { value } => value
                .ok_or("char type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<char>()
                        .map(EValue::Char)
                        .map_err(|_| format!("could not parse \"{}\" as char", v))
                }),
            crate::model::r#type::EType::Any => {
                Err("any-type cannot provide a specific value!".to_string())
            }
            crate::model::r#type::EType::None { message } => message
                .map(EValue::None)
                .ok_or("none type did not have a specific value!".to_string()),
        }
    }

    fn eval_value_for_function_call(
        function_declaration: &crate::model::function_declaration::FunctionDeclaration,
        arguments: Vec<EValue>,
    ) -> Result<EValue, String> {
        match (function_declaration.name.as_str(), arguments.as_slice()) {
            ("+", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Int(a.wrapping_add(*b))),
            ("/", [EValue::Int(a), EValue::Int(b)]) => {
                if *b == 0 {
                    Ok(EValue::None("division by zero".to_string()))
                } else {
                    Ok(EValue::Int(a.wrapping_div(*b)))
                }
            }
            ("charAt", [EValue::String(s), EValue::Int(i)]) => {
                match usize::try_from(*i).ok().and_then(|i| s.chars().nth(i)) {
                    Some(c) => Ok(EValue::Char(c)),
                    None => Ok(EValue::None(format!(
                        "charAt: index {} out of bounds for string of length {}",
                        i,
                        s.chars().count()
                    ))),
                }
            }
            ("*(-1)", [EValue::Int(a)]) => Ok(EValue::Int(a.wrapping_neg())),
            ("substr", [EValue::String(s), EValue::Int(begin), EValue::Int(length)]) => {
                let chars = s.chars().collect::<Vec<char>>();
                let start = usize::try_from(*begin).unwrap_or(0).min(chars.len());
                let end = start
                    .saturating_add(usize::try_from(*length).unwrap_or(0))
                    .min(chars.len());
                Ok(EValue::String(chars[start..end].iter().collect()))
            }
            (name @ ("+" | "/" | "charAt" | "*(-1)" | "substr"), args) => Err(format!(
                "function {} called with unexpected argument types ({})",
                name,
                args.iter()
                    .map(EValue::type_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            (name, _) => Err(format!("unknown function declaration {}", name)),
        }
    }

    fn eval_value_for_type_cast(
        input_value: EValue,
        target_type: crate::model::r#type::EType,
    ) -> EValue {
        match &target_type {
            crate::model::r#type::EType::Bool { value: Some(_) }
            | crate::model::r#type::EType::Int { value: Some(_) }
            | crate::model::r#type::EType::String { value: Some(_) }
            | crate::model::r#type::EType::Char { value: Some(_) } => {
                return Self::eval_value_for_type(target_type).unwrap_or_else(EValue::None);
            }
            _ => {}
        }

        if let EValue::None(message) = input_value {
            return EValue::None(message);
        }

        let input_type = input_value.type_name();
        match target_type {
            crate::model::r#type::EType::Bool { .. } => match input_value {
                EValue::Bool(b) => EValue::Bool(b),
                EValue::Int(i) => EValue::Bool(i != 0),
                EValue::String(s) => s.parse::<bool>().map(EValue::Bool).unwrap_or_else(|_| {
                    EValue::None(format!("cannot cast string \"{}\" to bool", s))
                }),
                _ => EValue::None(format!("cannot cast {} to bool", input_type)),
            },
            crate::model::r#type::EType::Int { .. } => match input_value {
                EValue::Bool(b) => EValue::Int(if b { 1 } else { 0 }),
                EValue::Int(i) => EValue::Int(i),
                EValue::Char(c) => EValue::Int(c as i32),
                EValue::String(s) => s.parse::<i32>().map(EValue::Int).unwrap_or_else(|_| {
                    EValue::None(format!("cannot cast string \"{}\" to int", s))
                }),
                _ => EValue::None(format!("cannot cast {} to int", input_type)),
            },
            crate::model::r#type::EType::String { .. } => EValue::String(match input_value {
                EValue::Bool(b) => b.to_string(),
                EValue::Int(i) => i.to_string(),
                EValue::String(s) => s,
                EValue::Char(c) => c.to_string(),
                EValue::None(message) => message,
            }),
            crate::model::r#type::EType::Char { .. } => match input_value {
                EValue::Char(c) => EValue::Char(c),
                EValue::Int(i) => u32::try_from(i)
                    .ok()
                    .and_then(char::from_u32)
                    .map(EValue::Char)
                    .unwrap_or_else(|| EValue::None(format!("cannot cast int {} to char", i))),
                EValue::String(s) => s.parse::<char>().map(EValue::Char).unwrap_or_else(|_| {
                    EValue::None(format!("cannot cast string \"{}\" to char", s))
                }),
                _ => EValue::None(format!("cannot cast {} to char", input_type)),
            },
            crate::model::r#type::EType::Any => input_value,
            crate::model::r#type::EType::None { message } => {
                EValue::None(message.unwrap_or_else(|| "none".to_string()))
            }
        }
    }

    /// True once the sink node carries a value, i.e. evaluation has reached the
    /// root and stepping further would not add anything. Used to grey out the
    /// `Next` button.
    pub fn is_evaluated(&self, ast: &crate::model::ast::Ast) -> bool {
        self.node_ids_to_values.contains_key(&ast.sink_node_id)
    }
}

impl std::fmt::Display for EValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EValue::Bool(value) => write!(f, "{}", value),
            EValue::Int(value) => write!(f, "{}", value),
            EValue::String(value) => write!(f, "{}", value),
            EValue::Char(value) => write!(f, "{}", value),
            EValue::None(_) => write!(f, "none"),
        }
    }
}

impl EValue {
    /// Parse a user-typed VarDecl input string into a value of the declared
    /// type. VarDecls carry their value in `user_vardecl_values` (their type
    /// literal is `None` at declaration time), so this is the path from the
    /// prompt modal into evaluation.
    pub fn parse(target: &crate::model::r#type::EType, raw: &str) -> Result<EValue, String> {
        match target {
            crate::model::r#type::EType::Bool { .. } => raw
                .parse::<bool>()
                .map(EValue::Bool)
                .map_err(|_| format!("could not parse \"{}\" as bool", raw)),
            crate::model::r#type::EType::Int { .. } => raw
                .parse::<i32>()
                .map(EValue::Int)
                .map_err(|_| format!("could not parse \"{}\" as int", raw)),
            crate::model::r#type::EType::String { .. } => Ok(EValue::String(raw.to_string())),
            crate::model::r#type::EType::Char { .. } => raw
                .parse::<char>()
                .map(EValue::Char)
                .map_err(|_| format!("could not parse \"{}\" as char", raw)),
            crate::model::r#type::EType::Any => {
                Err("any-typed var-decl needs a concrete type".to_string())
            }
            crate::model::r#type::EType::None { .. } => Ok(EValue::None(raw.to_string())),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            EValue::Bool(_) => "bool",
            EValue::Int(_) => "int",
            EValue::String(_) => "string",
            EValue::Char(_) => "char",
            EValue::None(_) => "none",
        }
    }
}
